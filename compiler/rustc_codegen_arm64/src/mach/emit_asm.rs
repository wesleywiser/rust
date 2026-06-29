//! Textual emission: turn a [`MachModule`] into Apple-flavoured AArch64 assembly (`--emit asm`).
//!
//! This mirrors the binary [`emit_obj`](crate::mach::emit_obj) path instruction-for-instruction;
//! the two are kept in sync and cross-checked by assembling the `.s` with `clang` and diffing the
//! bytes against our own encoder.

use std::fmt::Write;

use crate::mach::inst::{
    AddSub, AtomicRmwOp, CondSel, DataProc1, DataProc2, DmbOption, FpOp1, FpOp2, Inst, Label, LogicOp, MemSize,
    MovKind, PairIndex,
};
use crate::mach::module::{DataSection, MachModule};
use crate::mach::reg::{FpSize, OperandSize, Vreg};

/// Emit `module` as a textual assembly string.
pub fn emit_asm(module: &MachModule) -> String {
    let mut out = String::new();
    out.push_str("\t.section\t__TEXT,__text,regular,pure_instructions\n");

    for (fidx, f) in module.functions.iter().enumerate() {
        if f.is_global {
            let _ = writeln!(out, "\t.globl\t{}", f.name);
        }
        out.push_str("\t.p2align\t2\n");
        let _ = writeln!(out, "{}:", f.name);
        for inst in &f.insts {
            fmt_inst(&mut out, inst, fidx);
        }
    }

    for d in &module.data {
        // Thread-local data is emitted as a `$tlv$init` initializer in `__thread_data` plus the
        // 3-word `__thread_vars` descriptor (matching what the object writer synthesizes).
        if d.section == DataSection::Tls {
            out.push_str("\t.section\t__DATA,__thread_data,thread_local_regular\n");
            if d.align > 1 {
                let _ = writeln!(out, "\t.p2align\t{}", d.align.trailing_zeros());
            }
            let _ = writeln!(out, "{}$tlv$init:", d.name);
            emit_data_bytes(&mut out, &d.bytes, &d.relocs);
            out.push_str("\t.section\t__DATA,__thread_vars,thread_local_variables\n");
            if d.is_global {
                let _ = writeln!(out, "\t.globl\t{}", d.name);
            }
            let _ = writeln!(out, "{}:", d.name);
            let _ = writeln!(out, "\t.quad\t__tlv_bootstrap");
            let _ = writeln!(out, "\t.quad\t0");
            let _ = writeln!(out, "\t.quad\t{}$tlv$init", d.name);
            continue;
        }
        let directive = match d.section {
            DataSection::Data => "\t.section\t__DATA,__data\n",
            DataSection::ReadOnly => "\t.section\t__TEXT,__const\n",
            DataSection::Bss => "\t.section\t__DATA,__bss\n",
            DataSection::Tls => unreachable!("thread-local data handled above"),
        };
        out.push_str(directive);
        if d.is_global {
            let _ = writeln!(out, "\t.globl\t{}", d.name);
        }
        if d.align > 1 {
            let _ = writeln!(out, "\t.p2align\t{}", d.align.trailing_zeros());
        }
        let _ = writeln!(out, "{}:", d.name);
        match d.section {
            DataSection::Bss => {
                let _ = writeln!(out, "\t.zero\t{}", d.bss_size);
            }
            _ => emit_data_bytes(&mut out, &d.bytes, &d.relocs),
        }
    }

    out
}

/// Emit an initializer as `.byte` runs, with each pointer relocation rendered as a `.quad sym`
/// (`+addend`) directive in place of the eight bytes it covers. The relative offset that the binary
/// path keeps in the data (Mach-O implicit addend) is read back out here and made explicit.
fn emit_data_bytes(out: &mut String, bytes: &[u8], relocs: &[crate::mach::func::Reloc]) {
    use crate::mach::func::RelocKind;

    // Relocations in increasing offset order so we can walk the bytes once.
    let mut relocs: Vec<&crate::mach::func::Reloc> = relocs.iter().collect();
    relocs.sort_by_key(|r| r.offset);

    let mut pos: usize = 0;
    let mut ri = 0;
    while pos < bytes.len() {
        if ri < relocs.len() && relocs[ri].offset as usize == pos {
            let r = relocs[ri];
            debug_assert_eq!(r.kind, RelocKind::Unsigned64, "only 64-bit pointer relocs in data");
            // The implicit addend is the little-endian value currently stored at the pointer.
            let stored = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as i64;
            let addend = r.addend + stored;
            if addend != 0 {
                let _ = writeln!(out, "\t.quad\t{}+{}", r.sym, addend);
            } else {
                let _ = writeln!(out, "\t.quad\t{}", r.sym);
            }
            pos += 8;
            ri += 1;
        } else {
            // Emit raw bytes up to the next relocation (or the end of the data).
            let next = relocs.get(ri).map_or(bytes.len(), |r| r.offset as usize);
            for chunk in bytes[pos..next].chunks(16) {
                let line = chunk.iter().map(|b| format!("0x{b:02x}")).collect::<Vec<_>>().join(", ");
                let _ = writeln!(out, "\t.byte\t{line}");
            }
            pos = next;
        }
    }
}

/// Local label name for branch targets within a function.
fn label_name(fidx: usize, label: Label) -> String {
    format!("LBB{fidx}_{label}")
}

fn fmt_inst(out: &mut String, inst: &Inst, fidx: usize) {
    match *inst {
        Inst::Label(l) => {
            let _ = writeln!(out, "{}:", label_name(fidx, l));
        }
        Inst::Nop => line(out, "nop"),
        Inst::Brk { imm16 } => line(out, &format!("brk #{imm16}")),

        Inst::MovWide { kind, size, rd, imm16, shift } => {
            let op = match kind {
                MovKind::Zero => "movz",
                MovKind::Keep => "movk",
                MovKind::Inverse => "movn",
            };
            let rd = rd.name_zr(size);
            if shift == 0 {
                line(out, &format!("{op} {rd}, #{imm16}"));
            } else {
                line(out, &format!("{op} {rd}, #{imm16}, lsl #{shift}"));
            }
        }
        Inst::MovReg { size, rd, rm } => {
            line(out, &format!("mov {}, {}", rd.name_zr(size), rm.name_zr(size)));
        }
        Inst::MovSp { size, rd, rn } => {
            line(out, &format!("mov {}, {}", rd.name(size), rn.name(size)));
        }

        Inst::AddSubImm { op, size, set_flags, rd, rn, imm12, shift12 } => {
            let mnem = add_sub_mnem(op, set_flags);
            let shift = if shift12 { ", lsl #12" } else { "" };
            if set_flags && rd.encoding() == 31 {
                // cmp/cmn: destination is discarded.
                let cmp = if op == AddSub::Sub { "cmp" } else { "cmn" };
                line(out, &format!("{cmp} {}, #{imm12}{shift}", rn.name(size)));
            } else {
                line(
                    out,
                    &format!("{mnem} {}, {}, #{imm12}{shift}", rd.name(size), rn.name(size)),
                );
            }
        }
        Inst::AddSubReg { op, size, set_flags, rd, rn, rm, amount } => {
            let mnem = add_sub_mnem(op, set_flags);
            let shift = if amount == 0 { String::new() } else { format!(", lsl #{amount}") };
            if set_flags && rd.encoding() == 31 {
                let cmp = if op == AddSub::Sub { "cmp" } else { "cmn" };
                line(out, &format!("{cmp} {}, {}{shift}", rn.name_zr(size), rm.name_zr(size)));
            } else {
                line(
                    out,
                    &format!(
                        "{mnem} {}, {}, {}{shift}",
                        rd.name_zr(size),
                        rn.name_zr(size),
                        rm.name_zr(size)
                    ),
                );
            }
        }
        Inst::AddSubCarry { op, size, set_flags, rd, rn, rm } => {
            let mnem = match (op, set_flags) {
                (AddSub::Add, false) => "adc",
                (AddSub::Add, true) => "adcs",
                (AddSub::Sub, false) => "sbc",
                (AddSub::Sub, true) => "sbcs",
            };
            line(
                out,
                &format!(
                    "{mnem} {}, {}, {}",
                    rd.name_zr(size),
                    rn.name_zr(size),
                    rm.name_zr(size)
                ),
            );
        }
        Inst::AddSubExtReg { op, size, rd, rn, rm } => {
            let mnem = match op {
                AddSub::Add => "add",
                AddSub::Sub => "sub",
            };
            // `rd`/`rn` use the SP-aware name (register 31 is `sp`, not the zero register, in the
            // extended-register form).
            line(
                out,
                &format!("{mnem} {}, {}, {}", rd.name(size), rn.name(size), rm.name(size)),
            );
        }
        Inst::Logical { op, size, rd, rn, rm, amount } => {
            let mnem = match op {
                LogicOp::And => "and",
                LogicOp::Orr => "orr",
                LogicOp::Eor => "eor",
                LogicOp::Ands => "ands",
            };
            let shift = if amount == 0 { String::new() } else { format!(", lsl #{amount}") };
            line(
                out,
                &format!(
                    "{mnem} {}, {}, {}{shift}",
                    rd.name_zr(size),
                    rn.name_zr(size),
                    rm.name_zr(size)
                ),
            );
        }

        Inst::Madd { size, rd, rn, rm, ra } => {
            if ra.encoding() == 31 {
                line(out, &format!("mul {}, {}, {}", rd.name_zr(size), rn.name_zr(size), rm.name_zr(size)));
            } else {
                line(
                    out,
                    &format!(
                        "madd {}, {}, {}, {}",
                        rd.name_zr(size),
                        rn.name_zr(size),
                        rm.name_zr(size),
                        ra.name_zr(size)
                    ),
                );
            }
        }
        Inst::Msub { size, rd, rn, rm, ra } => {
            line(
                out,
                &format!(
                    "msub {}, {}, {}, {}",
                    rd.name_zr(size),
                    rn.name_zr(size),
                    rm.name_zr(size),
                    ra.name_zr(size)
                ),
            );
        }
        Inst::MulHigh { signed, rd, rn, rm } => {
            let mnem = if signed { "smulh" } else { "umulh" };
            line(
                out,
                &format!(
                    "{mnem} {}, {}, {}",
                    rd.name_zr(OperandSize::S64),
                    rn.name_zr(OperandSize::S64),
                    rm.name_zr(OperandSize::S64)
                ),
            );
        }
        Inst::DataProc2 { op, size, rd, rn, rm } => {
            let mnem = match op {
                DataProc2::Udiv => "udiv",
                DataProc2::Sdiv => "sdiv",
                DataProc2::Lslv => "lsl",
                DataProc2::Lsrv => "lsr",
                DataProc2::Asrv => "asr",
            };
            line(
                out,
                &format!("{mnem} {}, {}, {}", rd.name_zr(size), rn.name_zr(size), rm.name_zr(size)),
            );
        }

        Inst::DataProc1 { op, size, rd, rn } => {
            let mnem = match op {
                DataProc1::Clz => "clz",
                DataProc1::Rbit => "rbit",
                DataProc1::Rev => "rev",
            };
            line(out, &format!("{mnem} {}, {}", rd.name_zr(size), rn.name_zr(size)));
        }
        Inst::Sxt { from, to, rd, rn } => {
            let mnem = match from {
                MemSize::B => "sxtb",
                MemSize::H => "sxth",
                MemSize::W => "sxtw",
                MemSize::X => unreachable!("sxt from X is invalid"),
            };
            // The source is always the 32-bit (`W`) view of `rn`.
            line(out, &format!("{mnem} {}, {}", rd.name_zr(to), rn.name_zr(OperandSize::S32)));
        }
        Inst::CondSel { op, size, rd, rn, rm, cond } => {
            let mnem = match op {
                CondSel::Csel => "csel",
                CondSel::Csinc => "csinc",
                CondSel::Csinv => "csinv",
                CondSel::Csneg => "csneg",
            };
            line(
                out,
                &format!(
                    "{mnem} {}, {}, {}, {}",
                    rd.name_zr(size),
                    rn.name_zr(size),
                    rm.name_zr(size),
                    cond.mnemonic()
                ),
            );
        }

        Inst::LoadStoreUImm { load, signed, size, rt, rn, offset } => {
            let (mnem, rt_size) = load_store_mnem(load, signed, size);
            line(out, &format!("{mnem} {}, [{}, #{offset}]", rt.name_zr(rt_size), rn.name(OperandSize::S64)));
        }
        Inst::LoadStorePair { load, size, index, rt, rt2, rn, offset } => {
            let mnem = if load { "ldp" } else { "stp" };
            let rt = rt.name_zr(size);
            let rt2 = rt2.name_zr(size);
            let base = rn.name(OperandSize::S64);
            let text = match index {
                PairIndex::Offset => format!("{mnem} {rt}, {rt2}, [{base}, #{offset}]"),
                PairIndex::PreIndex => format!("{mnem} {rt}, {rt2}, [{base}, #{offset}]!"),
                PairIndex::PostIndex => format!("{mnem} {rt}, {rt2}, [{base}], #{offset}"),
            };
            line(out, &text);
        }

        Inst::B { target } => line(out, &format!("b {}", label_name(fidx, target))),
        Inst::BCond { cond, target } => {
            line(out, &format!("b.{} {}", cond.mnemonic(), label_name(fidx, target)))
        }
        Inst::CbNz { nonzero, size, rt, target } => {
            let mnem = if nonzero { "cbnz" } else { "cbz" };
            line(out, &format!("{mnem} {}, {}", rt.name_zr(size), label_name(fidx, target)));
        }

        Inst::Bl { ref sym } => line(out, &format!("bl {}", sym.name)),
        Inst::Blr { rn } => line(out, &format!("blr {}", rn.name(OperandSize::S64))),
        Inst::Br { rn } => line(out, &format!("br {}", rn.name(OperandSize::S64))),
        Inst::Ret { rn } => line(out, &format!("ret {}", rn.name(OperandSize::S64))),

        Inst::Adrp { rd, ref sym } => {
            line(out, &format!("adrp {}, {}@PAGE", rd.name(OperandSize::S64), sym.name))
        }
        Inst::AdrpTlv { rd, ref sym } => {
            line(out, &format!("adrp {}, {}@TLVPPAGE", rd.name(OperandSize::S64), sym.name))
        }
        Inst::LdrTlvLo { rt, rn, ref sym } => line(
            out,
            &format!(
                "ldr {}, [{}, {}@TLVPPAGEOFF]",
                rt.name(OperandSize::S64),
                rn.name(OperandSize::S64),
                sym.name
            ),
        ),
        Inst::AdrpGot { rd, ref sym } => {
            line(out, &format!("adrp {}, {}@GOTPAGE", rd.name(OperandSize::S64), sym.name))
        }
        Inst::LdrGotLo { rt, rn, ref sym } => line(
            out,
            &format!(
                "ldr {}, [{}, {}@GOTPAGEOFF]",
                rt.name(OperandSize::S64),
                rn.name(OperandSize::S64),
                sym.name
            ),
        ),
        Inst::AddLo { rd, rn, ref sym } => line(
            out,
            &format!(
                "add {}, {}, {}@PAGEOFF",
                rd.name(OperandSize::S64),
                rn.name(OperandSize::S64),
                sym.name
            ),
        ),

        Inst::FpDataProc2 { op, size, rd, rn, rm } => {
            let mnem = match op {
                FpOp2::Fadd => "fadd",
                FpOp2::Fsub => "fsub",
                FpOp2::Fmul => "fmul",
                FpOp2::Fdiv => "fdiv",
            };
            line(out, &format!("{mnem} {}, {}, {}", vname(rd, size), vname(rn, size), vname(rm, size)));
        }
        Inst::FpDataProc1 { op, size, rd, rn } => {
            let mnem = match op {
                FpOp1::Fabs => "fabs",
                FpOp1::Fneg => "fneg",
                FpOp1::Fsqrt => "fsqrt",
                FpOp1::Frintn => "frintn",
                FpOp1::Frintp => "frintp",
                FpOp1::Frintm => "frintm",
                FpOp1::Frintz => "frintz",
                FpOp1::Frinta => "frinta",
            };
            line(out, &format!("{mnem} {}, {}", vname(rd, size), vname(rn, size)));
        }
        Inst::FpFma { size, rd, rn, rm, ra } => {
            line(
                out,
                &format!(
                    "fmadd {}, {}, {}, {}",
                    vname(rd, size),
                    vname(rn, size),
                    vname(rm, size),
                    vname(ra, size)
                ),
            );
        }
        Inst::FmovFromGpr { size, rd, rn } => {
            let gpr_size = match size {
                FpSize::S32 => OperandSize::S32,
                FpSize::S64 => OperandSize::S64,
            };
            line(out, &format!("fmov {}, {}", vname(rd, size), rn.name_zr(gpr_size)));
        }
        Inst::LoadStoreFpUImm { load, size, rt, rn, offset } => {
            let mnem = if load { "ldr" } else { "str" };
            line(
                out,
                &format!("{mnem} {}, [{}, #{offset}]", vname(rt, size), rn.name(OperandSize::S64)),
            );
        }
        Inst::FpCmp { size, rn, rm } => {
            line(out, &format!("fcmp {}, {}", vname(rn, size), vname(rm, size)));
        }
        Inst::IntToFp { signed, fp, int, rd, rn } => {
            let mnem = if signed { "scvtf" } else { "ucvtf" };
            line(out, &format!("{mnem} {}, {}", vname(rd, fp), rn.name_zr(int)));
        }
        Inst::FpToInt { signed, fp, int, rd, rn } => {
            let mnem = if signed { "fcvtzs" } else { "fcvtzu" };
            line(out, &format!("{mnem} {}, {}", rd.name_zr(int), vname(rn, fp)));
        }
        Inst::FpCvt { from, to, rd, rn } => {
            line(out, &format!("fcvt {}, {}", vname(rd, to), vname(rn, from)));
        }

        Inst::LoadAcq { size, rt, rn } => {
            line(
                out,
                &format!(
                    "ldar{} {}, [{}]",
                    mem_suffix(size),
                    rt.name_zr(gpr_size(size)),
                    rn.name(OperandSize::S64)
                ),
            );
        }
        Inst::StoreRel { size, rt, rn } => {
            line(
                out,
                &format!(
                    "stlr{} {}, [{}]",
                    mem_suffix(size),
                    rt.name_zr(gpr_size(size)),
                    rn.name(OperandSize::S64)
                ),
            );
        }
        Inst::AtomicRmw { op, acquire, release, size, rs, rt, rn } => {
            let base = match op {
                AtomicRmwOp::Add => "ldadd",
                AtomicRmwOp::Clr => "ldclr",
                AtomicRmwOp::Eor => "ldeor",
                AtomicRmwOp::Set => "ldset",
                AtomicRmwOp::Smax => "ldsmax",
                AtomicRmwOp::Smin => "ldsmin",
                AtomicRmwOp::Umax => "ldumax",
                AtomicRmwOp::Umin => "ldumin",
                AtomicRmwOp::Swp => "swp",
            };
            let gs = gpr_size(size);
            line(
                out,
                &format!(
                    "{base}{}{} {}, {}, [{}]",
                    ar_suffix(acquire, release),
                    mem_suffix(size),
                    rs.name_zr(gs),
                    rt.name_zr(gs),
                    rn.name(OperandSize::S64)
                ),
            );
        }
        Inst::AtomicCas { acquire, release, size, rs, rt, rn } => {
            let gs = gpr_size(size);
            line(
                out,
                &format!(
                    "cas{}{} {}, {}, [{}]",
                    ar_suffix(acquire, release),
                    mem_suffix(size),
                    rs.name_zr(gs),
                    rt.name_zr(gs),
                    rn.name(OperandSize::S64)
                ),
            );
        }
        Inst::Dmb { option } => {
            let opt = match option {
                DmbOption::Ish => "ish",
                DmbOption::IshLd => "ishld",
                DmbOption::IshSt => "ishst",
            };
            line(out, &format!("dmb {opt}"));
        }
        Inst::Isb => line(out, "isb"),
    }
}

/// The `b`/`h` mnemonic suffix for a byte/halfword atomic (word/doubleword take none).
fn mem_suffix(size: MemSize) -> &'static str {
    match size {
        MemSize::B => "b",
        MemSize::H => "h",
        MemSize::W | MemSize::X => "",
    }
}

/// The `a`/`l`/`al` ordering suffix from the acquire/release bits.
fn ar_suffix(acquire: bool, release: bool) -> &'static str {
    match (acquire, release) {
        (true, true) => "al",
        (true, false) => "a",
        (false, true) => "l",
        (false, false) => "",
    }
}

/// The GPR view for an atomic of the given access size (`x` for doublewords, otherwise `w`).
fn gpr_size(size: MemSize) -> OperandSize {
    match size {
        MemSize::X => OperandSize::S64,
        MemSize::B | MemSize::H | MemSize::W => OperandSize::S32,
    }
}

fn line(out: &mut String, body: &str) {
    out.push('\t');
    out.push_str(body);
    out.push('\n');
}

fn add_sub_mnem(op: AddSub, set_flags: bool) -> &'static str {
    match (op, set_flags) {
        (AddSub::Add, false) => "add",
        (AddSub::Add, true) => "adds",
        (AddSub::Sub, false) => "sub",
        (AddSub::Sub, true) => "subs",
    }
}

/// Mnemonic and register width for a load/store of the given size/signedness.
fn load_store_mnem(load: bool, signed: bool, size: MemSize) -> (&'static str, OperandSize) {
    match (load, signed, size) {
        (false, _, MemSize::B) => ("strb", OperandSize::S32),
        (false, _, MemSize::H) => ("strh", OperandSize::S32),
        (false, _, MemSize::W) => ("str", OperandSize::S32),
        (false, _, MemSize::X) => ("str", OperandSize::S64),
        (true, false, MemSize::B) => ("ldrb", OperandSize::S32),
        (true, false, MemSize::H) => ("ldrh", OperandSize::S32),
        (true, false, MemSize::W) => ("ldr", OperandSize::S32),
        (true, false, MemSize::X) => ("ldr", OperandSize::S64),
        (true, true, MemSize::B) => ("ldrsb", OperandSize::S64),
        (true, true, MemSize::H) => ("ldrsh", OperandSize::S64),
        (true, true, MemSize::W) => ("ldrsw", OperandSize::S64),
        (true, true, MemSize::X) => ("ldr", OperandSize::S64),
    }
}

fn vname(v: Vreg, size: FpSize) -> String {
    v.name(size)
}
