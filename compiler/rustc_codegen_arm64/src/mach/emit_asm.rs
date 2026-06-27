//! Textual emission: turn a [`MachModule`] into Apple-flavoured AArch64 assembly (`--emit asm`).
//!
//! This mirrors the binary [`emit_obj`](crate::mach::emit_obj) path instruction-for-instruction;
//! the two are kept in sync and cross-checked by assembling the `.s` with `clang` and diffing the
//! bytes against our own encoder.

use std::fmt::Write;

use crate::mach::inst::{
    AddSub, CondSel, DataProc2, FpOp2, Inst, Label, LogicOp, MemSize, MovKind, PairIndex,
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
        let directive = match d.section {
            DataSection::Data => "\t.section\t__DATA,__data\n",
            DataSection::ReadOnly => "\t.section\t__TEXT,__const\n",
            DataSection::Bss => "\t.section\t__DATA,__bss\n",
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
            _ => {
                // FIXME: data with embedded relocations is not yet rendered textually; the binary
                // object path handles those. Plain initializer bytes are emitted here.
                for chunk in d.bytes.chunks(16) {
                    let bytes =
                        chunk.iter().map(|b| format!("0x{b:02x}")).collect::<Vec<_>>().join(", ");
                    let _ = writeln!(out, "\t.byte\t{bytes}");
                }
            }
        }
    }

    out
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
