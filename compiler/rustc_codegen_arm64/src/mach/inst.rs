//! The AArch64 instruction model and its binary (`u32`) encoder.
//!
//! Each [`Inst`] lowers to either a 32-bit machine word (via [`Inst::encode`], used for object
//! emission) or textual assembly (see `emit_asm`). Instructions that reference labels or symbols
//! encode with zeroed immediate fields; the containing function patches branch displacements and
//! records relocations during layout.

use crate::mach::reg::{Cond, FpSize, Gpr, OperandSize, Vreg};

/// Index of a local label within a function.
pub type Label = u32;

/// A reference to a named symbol plus an addend, used by calls and address-formation sequences.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SymRef {
    pub name: Box<str>,
    pub addend: i64,
}

impl SymRef {
    pub fn new(name: impl Into<Box<str>>) -> SymRef {
        SymRef { name: name.into(), addend: 0 }
    }
}

/// Which wide-immediate move to emit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MovKind {
    /// `movz`: zero the register, then insert the immediate at `shift`.
    Zero,
    /// `movk`: keep other bits, overwrite the 16-bit field at `shift`.
    Keep,
    /// `movn`: insert the bitwise-inverted immediate.
    Inverse,
}

/// Add/sub opcode selector.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AddSub {
    Add,
    Sub,
}

/// Bitwise logical opcode selector (shifted-register forms).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LogicOp {
    And,
    Orr,
    Eor,
    Ands,
}

/// Two-source data-processing opcode (division / variable shifts).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DataProc2 {
    Udiv,
    Sdiv,
    Lslv,
    Lsrv,
    Asrv,
}

/// One-source data-processing opcode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DataProc1 {
    /// `clz` — count leading zeros.
    Clz,
    /// `rbit` — reverse bit order.
    Rbit,
    /// `rev` — reverse byte order (full register width).
    Rev,
}

/// Conditional-select opcode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CondSel {
    /// `csel rd, rn, rm, cond`
    Csel,
    /// `csinc rd, rn, rm, cond` (basis for `cset`/`csinc`)
    Csinc,
    /// `csinv rd, rn, rm, cond`
    Csinv,
    /// `csneg rd, rn, rm, cond`
    Csneg,
}

/// Width / signedness of a memory access.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MemSize {
    /// 1 byte.
    B,
    /// 2 bytes.
    H,
    /// 4 bytes.
    W,
    /// 8 bytes.
    X,
}

impl MemSize {
    /// The `size` field (bits 31:30) of a load/store.
    const fn size_field(self) -> u32 {
        match self {
            MemSize::B => 0b00,
            MemSize::H => 0b01,
            MemSize::W => 0b10,
            MemSize::X => 0b11,
        }
    }

    /// Access width in bytes (the scale for unsigned-offset immediates).
    pub const fn bytes(self) -> u32 {
        match self {
            MemSize::B => 1,
            MemSize::H => 2,
            MemSize::W => 4,
            MemSize::X => 8,
        }
    }
}

/// Indexing mode for the load/store-pair instructions used in prologues/epilogues.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PairIndex {
    /// `[rn, #imm]`
    Offset,
    /// `[rn, #imm]!`
    PreIndex,
    /// `[rn], #imm`
    PostIndex,
}

/// A single AArch64 instruction (or a label pseudo-instruction).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Inst {
    /// A local label definition; emits no bytes.
    Label(Label),

    /// `nop`
    Nop,
    /// `brk #imm16`
    Brk { imm16: u16 },

    /// `movz`/`movk`/`movn rd, #imm16, lsl #shift`. `shift` is one of 0/16/32/48.
    MovWide { kind: MovKind, size: OperandSize, rd: Gpr, imm16: u16, shift: u8 },
    /// `orr rd, zr, rm` — register-to-register move.
    MovReg { size: OperandSize, rd: Gpr, rm: Gpr },
    /// `mov rd, sp` / `mov sp, rn` — `add rd, rn, #0` form (SP-aware move).
    MovSp { size: OperandSize, rd: Gpr, rn: Gpr },

    /// `add`/`sub`(`s`) `rd, rn, #imm12 [, lsl #12]`.
    AddSubImm {
        op: AddSub,
        size: OperandSize,
        set_flags: bool,
        rd: Gpr,
        rn: Gpr,
        imm12: u16,
        shift12: bool,
    },
    /// `add`/`sub`(`s`) `rd, rn, rm` (shifted-register form, LSL by `amount`).
    AddSubReg {
        op: AddSub,
        size: OperandSize,
        set_flags: bool,
        rd: Gpr,
        rn: Gpr,
        rm: Gpr,
        amount: u8,
    },
    /// `and`/`orr`/`eor`/`ands rd, rn, rm` (shifted-register form).
    Logical { op: LogicOp, size: OperandSize, rd: Gpr, rn: Gpr, rm: Gpr, amount: u8 },

    /// `madd rd, rn, rm, ra` (`mul` is `madd … , zr`).
    Madd { size: OperandSize, rd: Gpr, rn: Gpr, rm: Gpr, ra: Gpr },
    /// `msub rd, rn, rm, ra` (basis for `rem`).
    Msub { size: OperandSize, rd: Gpr, rn: Gpr, rm: Gpr, ra: Gpr },
    /// `umulh`/`smulh rd, rn, rm` — the high 64 bits of a 64x64-bit multiply.
    MulHigh { signed: bool, rd: Gpr, rn: Gpr, rm: Gpr },
    /// `udiv`/`sdiv`/`lslv`/`lsrv`/`asrv rd, rn, rm`.
    DataProc2 { op: DataProc2, size: OperandSize, rd: Gpr, rn: Gpr, rm: Gpr },

    /// `clz rd, rn` (one-source data-processing).
    DataProc1 { op: DataProc1, size: OperandSize, rd: Gpr, rn: Gpr },

    /// `sxtb`/`sxth`/`sxtw rd, wn` — sign-extend the low `from` bits of `rn` to `to` bits (an
    /// `sbfm` alias). The source is always read as a 32-bit (`W`) register.
    Sxt { from: MemSize, to: OperandSize, rd: Gpr, rn: Gpr },

    /// `csel`/`csinc`/`csinv`/`csneg rd, rn, rm, cond`.
    CondSel { op: CondSel, size: OperandSize, rd: Gpr, rn: Gpr, rm: Gpr, cond: Cond },

    /// Load/store with an unsigned, scaled 12-bit immediate offset: `[rn, #imm]`.
    LoadStoreUImm { load: bool, signed: bool, size: MemSize, rt: Gpr, rn: Gpr, offset: u32 },
    /// Load/store pair: `stp`/`ldp rt, rt2, [rn{, #imm}]` with the given index mode.
    LoadStorePair {
        load: bool,
        size: OperandSize,
        index: PairIndex,
        rt: Gpr,
        rt2: Gpr,
        rn: Gpr,
        /// Byte offset; must be a multiple of the access size.
        offset: i32,
    },

    /// `b <label>`
    B { target: Label },
    /// `b.<cond> <label>`
    BCond { cond: Cond, target: Label },
    /// `cbz`/`cbnz rt, <label>`
    CbNz { nonzero: bool, size: OperandSize, rt: Gpr, target: Label },

    /// `bl <sym>` — direct call (records a `Branch26` relocation).
    Bl { sym: SymRef },
    /// `blr rn` — indirect call.
    Blr { rn: Gpr },
    /// `br rn` — indirect tail branch.
    Br { rn: Gpr },
    /// `ret {rn}` — defaults to `x30`.
    Ret { rn: Gpr },

    /// `adrp rd, <sym>@PAGE` — records a `Page21` relocation.
    Adrp { rd: Gpr, sym: SymRef },
    /// `add rd, rn, #:lo12:<sym>` — records a `PageOff12` relocation (the `@PAGEOFF` add).
    AddLo { rd: Gpr, rn: Gpr, sym: SymRef },

    /// `fadd`/`fsub`/`fmul`/`fdiv` two-operand FP.
    FpDataProc2 { op: FpOp2, size: FpSize, rd: Vreg, rn: Vreg, rm: Vreg },
    /// `fneg`/`fabs`/`fsqrt` one-operand FP.
    FpDataProc1 { op: FpOp1, size: FpSize, rd: Vreg, rn: Vreg },
    /// `fmov rd, rn` moving a general register's raw bits into a scalar FP register (used to
    /// materialize floating-point constants).
    FmovFromGpr { size: FpSize, rd: Vreg, rn: Gpr },
    /// `ldr`/`str` of a scalar FP register with an unsigned, scaled 12-bit immediate offset.
    LoadStoreFpUImm { load: bool, size: FpSize, rt: Vreg, rn: Gpr, offset: u32 },
    /// `fcmp rn, rm` — floating-point compare, setting the NZCV flags.
    FpCmp { size: FpSize, rn: Vreg, rm: Vreg },
    /// `scvtf`/`ucvtf` — integer (in a GPR) to floating-point conversion.
    IntToFp { signed: bool, fp: FpSize, int: OperandSize, rd: Vreg, rn: Gpr },
    /// `fcvtzs`/`fcvtzu` — floating-point to integer, rounding toward zero (saturating in hardware).
    FpToInt { signed: bool, fp: FpSize, int: OperandSize, rd: Gpr, rn: Vreg },
    /// `fcvt` — floating-point precision conversion (`f32`<->`f64`).
    FpCvt { from: FpSize, to: FpSize, rd: Vreg, rn: Vreg },

    /// `ldar`/`ldarb`/`ldarh` — load-acquire (atomic acquire load).
    LoadAcq { size: MemSize, rt: Gpr, rn: Gpr },
    /// `stlr`/`stlrb`/`stlrh` — store-release (atomic release store).
    StoreRel { size: MemSize, rt: Gpr, rn: Gpr },
    /// LSE atomic read-modify-write: `rt` receives the old `[rn]`; `[rn]` becomes `op(old, rs)`.
    /// `acquire`/`release` set the `A`/`R` ordering bits.
    AtomicRmw {
        op: AtomicRmwOp,
        acquire: bool,
        release: bool,
        size: MemSize,
        rs: Gpr,
        rt: Gpr,
        rn: Gpr,
    },
    /// `cas` — compare-and-swap: if `[rn] == rs` then `[rn] <- rt`; `rs` receives the old `[rn]`.
    AtomicCas { acquire: bool, release: bool, size: MemSize, rs: Gpr, rt: Gpr, rn: Gpr },
    /// `dmb` — data memory barrier.
    Dmb { option: DmbOption },
}

/// LSE atomic read-modify-write opcode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AtomicRmwOp {
    /// `ldadd` — `mem += rs`.
    Add,
    /// `ldclr` — `mem &= ~rs`.
    Clr,
    /// `ldeor` — `mem ^= rs`.
    Eor,
    /// `ldset` — `mem |= rs`.
    Set,
    /// `ldsmax` — signed max.
    Smax,
    /// `ldsmin` — signed min.
    Smin,
    /// `ldumax` — unsigned max.
    Umax,
    /// `ldumin` — unsigned min.
    Umin,
    /// `swp` — exchange (`mem <- rs`).
    Swp,
}

/// Shareability/domain option for the `dmb` barrier.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DmbOption {
    /// `ish` — full inner-shareable barrier.
    Ish,
    /// `ishld` — inner-shareable load barrier.
    IshLd,
    /// `ishst` — inner-shareable store barrier.
    IshSt,
}

/// Two-operand floating-point opcode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FpOp2 {
    Fadd,
    Fsub,
    Fmul,
    Fdiv,
}

/// One-operand floating-point opcode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FpOp1 {
    Fabs,
    Fneg,
    Fsqrt,
}

impl Inst {
    /// Encode to a 32-bit little-endian instruction word.
    ///
    /// Label-relative branches and symbol references encode with zeroed immediate fields; the
    /// containing function patches branch displacements and emits relocations during layout.
    pub fn encode(&self) -> u32 {
        match *self {
            Inst::Label(_) => panic!("Inst::Label has no encoding; resolve labels during layout"),

            Inst::Nop => 0xD503201F,
            Inst::Brk { imm16 } => 0xD4200000 | ((imm16 as u32) << 5),

            Inst::MovWide { kind, size, rd, imm16, shift } => {
                let opc: u32 = match kind {
                    MovKind::Inverse => 0b00,
                    MovKind::Zero => 0b10,
                    MovKind::Keep => 0b11,
                };
                let hw = (shift / 16) as u32;
                (size.sf() << 31)
                    | (opc << 29)
                    | (0b100101 << 23)
                    | (hw << 21)
                    | ((imm16 as u32) << 5)
                    | rd.encoding()
            }
            Inst::MovReg { size, rd, rm } => {
                // orr rd, zr, rm
                encode_logical(LogicOp::Orr, size, rd, Gpr::from_encoding(31), rm, 0)
            }
            Inst::MovSp { size, rd, rn } => {
                // add rd, rn, #0  (SP-form move)
                encode_add_sub_imm(AddSub::Add, size, false, rd, rn, 0, false)
            }

            Inst::AddSubImm { op, size, set_flags, rd, rn, imm12, shift12 } => {
                encode_add_sub_imm(op, size, set_flags, rd, rn, imm12, shift12)
            }
            Inst::AddSubReg { op, size, set_flags, rd, rn, rm, amount } => {
                let op_bit = match op {
                    AddSub::Add => 0,
                    AddSub::Sub => 1,
                };
                (size.sf() << 31)
                    | (op_bit << 30)
                    | ((set_flags as u32) << 29)
                    | (0b01011 << 24)
                    | (rm.encoding() << 16)
                    | ((amount as u32) << 10)
                    | (rn.encoding() << 5)
                    | rd.encoding()
            }
            Inst::Logical { op, size, rd, rn, rm, amount } => {
                encode_logical(op, size, rd, rn, rm, amount)
            }

            Inst::Madd { size, rd, rn, rm, ra } => {
                (size.sf() << 31)
                    | (0b11011 << 24)
                    | (rm.encoding() << 16)
                    | (ra.encoding() << 10)
                    | (rn.encoding() << 5)
                    | rd.encoding()
            }
            Inst::Msub { size, rd, rn, rm, ra } => {
                (size.sf() << 31)
                    | (0b11011 << 24)
                    | (rm.encoding() << 16)
                    | (1 << 15)
                    | (ra.encoding() << 10)
                    | (rn.encoding() << 5)
                    | rd.encoding()
            }
            Inst::MulHigh { signed, rd, rn, rm } => {
                let base: u32 = if signed { 0x9B40_7C00 } else { 0x9BC0_7C00 };
                base | (rm.encoding() << 16) | (rn.encoding() << 5) | rd.encoding()
            }
            Inst::DataProc2 { op, size, rd, rn, rm } => {
                let opcode: u32 = match op {
                    DataProc2::Udiv => 0b000010,
                    DataProc2::Sdiv => 0b000011,
                    DataProc2::Lslv => 0b001000,
                    DataProc2::Lsrv => 0b001001,
                    DataProc2::Asrv => 0b001010,
                };
                (size.sf() << 31)
                    | (0b11010110 << 21)
                    | (rm.encoding() << 16)
                    | (opcode << 10)
                    | (rn.encoding() << 5)
                    | rd.encoding()
            }

            Inst::DataProc1 { op, size, rd, rn } => {
                let opcode: u32 = match op {
                    DataProc1::Clz => 0b000100,
                    DataProc1::Rbit => 0b000000,
                    // Full-width byte reverse: `rev` (32-bit) vs `rev` (64-bit, aka REV64) differ
                    // in the opcode field.
                    DataProc1::Rev => match size {
                        OperandSize::S32 => 0b000010,
                        OperandSize::S64 => 0b000011,
                    },
                };
                (size.sf() << 31)
                    | (0b1 << 30)
                    | (0b11010110 << 21)
                    | (opcode << 10)
                    | (rn.encoding() << 5)
                    | rd.encoding()
            }

            Inst::Sxt { from, to, rd, rn } => {
                // `sbfm rd, rn, #0, #(width-1)` with the sign-extending opcode.
                let (sf, n): (u32, u32) = match to {
                    OperandSize::S32 => (0, 0),
                    OperandSize::S64 => (1, 1),
                };
                let imms: u32 = match from {
                    MemSize::B => 7,
                    MemSize::H => 15,
                    MemSize::W => 31,
                    MemSize::X => panic!("sxt cannot sign-extend from a 64-bit source"),
                };
                (sf << 31)
                    | (0b100110 << 23)
                    | (n << 22)
                    | (imms << 10)
                    | (rn.encoding() << 5)
                    | rd.encoding()
            }

            Inst::CondSel { op, size, rd, rn, rm, cond } => {
                let (op_bit, op2) = match op {
                    CondSel::Csel => (0, 0b00),
                    CondSel::Csinc => (0, 0b01),
                    CondSel::Csinv => (1, 0b00),
                    CondSel::Csneg => (1, 0b01),
                };
                (size.sf() << 31)
                    | (op_bit << 30)
                    | (0b11010100 << 21)
                    | (rm.encoding() << 16)
                    | (cond.encoding() << 12)
                    | (op2 << 10)
                    | (rn.encoding() << 5)
                    | rd.encoding()
            }

            Inst::LoadStoreUImm { load, signed, size, rt, rn, offset } => {
                let opc: u32 = if !load {
                    0b00
                } else if !signed {
                    0b01
                } else if size == MemSize::W {
                    // signed 32-bit load extends to 64-bit
                    0b10
                } else {
                    0b10
                };
                let scaled = offset / size.bytes();
                debug_assert!(offset % size.bytes() == 0, "unaligned scaled offset");
                debug_assert!(scaled < (1 << 12), "offset out of range for unsigned-imm form");
                (size.size_field() << 30)
                    | (0b111 << 27)
                    | (0b01 << 24)
                    | (opc << 22)
                    | (scaled << 10)
                    | (rn.encoding() << 5)
                    | rt.encoding()
            }
            Inst::LoadStorePair { load, size, index, rt, rt2, rn, offset } => {
                let opc: u32 = match size {
                    OperandSize::S32 => 0b00,
                    OperandSize::S64 => 0b10,
                };
                let index_bits: u32 = match index {
                    PairIndex::PostIndex => 0b001,
                    PairIndex::Offset => 0b010,
                    PairIndex::PreIndex => 0b011,
                };
                let scale = size.bytes() as i32;
                debug_assert!(offset % scale == 0, "unaligned pair offset");
                let imm7 = ((offset / scale) as u32) & 0x7F;
                (opc << 30)
                    | (0b101 << 27)
                    | (index_bits << 23)
                    | ((load as u32) << 22)
                    | (imm7 << 15)
                    | (rt2.encoding() << 10)
                    | (rn.encoding() << 5)
                    | rt.encoding()
            }

            // Label-relative branches encode with a zero displacement; patched during layout.
            Inst::B { .. } => 0x14000000,
            Inst::BCond { cond, .. } => 0x54000000 | cond.encoding(),
            Inst::CbNz { nonzero, size, rt, .. } => {
                (size.sf() << 31) | (0b011010 << 25) | ((nonzero as u32) << 24) | rt.encoding()
            }

            // `bl` encodes with a zero displacement; a Branch26 relocation supplies the target.
            Inst::Bl { .. } => 0x94000000,
            Inst::Blr { rn } => 0xD63F0000 | (rn.encoding() << 5),
            Inst::Br { rn } => 0xD61F0000 | (rn.encoding() << 5),
            Inst::Ret { rn } => 0xD65F0000 | (rn.encoding() << 5),

            // adrp/add encode with zeroed immediates; Page21/PageOff12 relocations patch them.
            Inst::Adrp { rd, .. } => 0x90000000 | rd.encoding(),
            Inst::AddLo { rd, rn, .. } => {
                encode_add_sub_imm(AddSub::Add, OperandSize::S64, false, rd, rn, 0, false)
            }

            Inst::FpDataProc2 { op, size, rd, rn, rm } => {
                let opcode: u32 = match op {
                    FpOp2::Fmul => 0b0000,
                    FpOp2::Fdiv => 0b0001,
                    FpOp2::Fadd => 0b0010,
                    FpOp2::Fsub => 0b0011,
                };
                (0b11110 << 24)
                    | (size.ftype() << 22)
                    | (1 << 21)
                    | (rm.encoding() << 16)
                    | (opcode << 12)
                    | (0b10 << 10)
                    | (rn.encoding() << 5)
                    | rd.encoding()
            }
            Inst::FpDataProc1 { op, size, rd, rn } => {
                let opcode: u32 = match op {
                    FpOp1::Fabs => 0b000001,
                    FpOp1::Fneg => 0b000010,
                    FpOp1::Fsqrt => 0b000011,
                };
                (0b00011110 << 24)
                    | (size.ftype() << 22)
                    | (1 << 21)
                    | (opcode << 15)
                    | (0b10000 << 10)
                    | (rn.encoding() << 5)
                    | rd.encoding()
            }
            Inst::FmovFromGpr { size, rd, rn } => {
                let base: u32 = match size {
                    FpSize::S64 => 0x9E67_0000,
                    FpSize::S32 => 0x1E27_0000,
                };
                base | (rn.encoding() << 5) | rd.encoding()
            }
            Inst::LoadStoreFpUImm { load, size, rt, rn, offset } => {
                let base: u32 = match (load, size) {
                    (false, FpSize::S64) => 0xFD00_0000,
                    (true, FpSize::S64) => 0xFD40_0000,
                    (false, FpSize::S32) => 0xBD00_0000,
                    (true, FpSize::S32) => 0xBD40_0000,
                };
                let scale = match size {
                    FpSize::S32 => 4,
                    FpSize::S64 => 8,
                };
                debug_assert!(offset % scale == 0, "unaligned scaled FP offset");
                let scaled = offset / scale;
                debug_assert!(scaled < (1 << 12), "FP offset out of range for unsigned-imm form");
                base | (scaled << 10) | (rn.encoding() << 5) | rt.encoding()
            }
            Inst::FpCmp { size, rn, rm } => {
                let base: u32 = match size {
                    FpSize::S64 => 0x1E60_2000,
                    FpSize::S32 => 0x1E20_2000,
                };
                base | (rm.encoding() << 16) | (rn.encoding() << 5)
            }
            Inst::IntToFp { signed, fp, int, rd, rn } => {
                let opcode: u32 = if signed { 0b010 } else { 0b011 };
                (int.sf() << 31)
                    | (0b11110 << 24)
                    | (fp.ftype() << 22)
                    | (1 << 21)
                    | (opcode << 16)
                    | (rn.encoding() << 5)
                    | rd.encoding()
            }
            Inst::FpToInt { signed, fp, int, rd, rn } => {
                let opcode: u32 = if signed { 0b000 } else { 0b001 };
                (int.sf() << 31)
                    | (0b11110 << 24)
                    | (fp.ftype() << 22)
                    | (1 << 21)
                    | (0b11 << 19)
                    | (opcode << 16)
                    | (rn.encoding() << 5)
                    | rd.encoding()
            }
            Inst::FpCvt { from, to, rd, rn } => {
                let base: u32 = match (from, to) {
                    (FpSize::S64, FpSize::S32) => 0x1E62_4000,
                    (FpSize::S32, FpSize::S64) => 0x1E22_C000,
                    _ => panic!("unsupported fcvt {from:?} -> {to:?}"),
                };
                base | (rn.encoding() << 5) | rd.encoding()
            }

            Inst::LoadAcq { size, rt, rn } => {
                (size.size_field() << 30) | 0x08DF_FC00 | (rn.encoding() << 5) | rt.encoding()
            }
            Inst::StoreRel { size, rt, rn } => {
                (size.size_field() << 30) | 0x089F_FC00 | (rn.encoding() << 5) | rt.encoding()
            }
            Inst::AtomicRmw { op, acquire, release, size, rs, rt, rn } => {
                let (o3, opc): (u32, u32) = match op {
                    AtomicRmwOp::Add => (0, 0b000),
                    AtomicRmwOp::Clr => (0, 0b001),
                    AtomicRmwOp::Eor => (0, 0b010),
                    AtomicRmwOp::Set => (0, 0b011),
                    AtomicRmwOp::Smax => (0, 0b100),
                    AtomicRmwOp::Smin => (0, 0b101),
                    AtomicRmwOp::Umax => (0, 0b110),
                    AtomicRmwOp::Umin => (0, 0b111),
                    AtomicRmwOp::Swp => (1, 0b000),
                };
                (size.size_field() << 30)
                    | 0x3800_0000
                    | ((acquire as u32) << 23)
                    | ((release as u32) << 22)
                    | (1 << 21)
                    | (rs.encoding() << 16)
                    | (o3 << 15)
                    | (opc << 12)
                    | (rn.encoding() << 5)
                    | rt.encoding()
            }
            Inst::AtomicCas { acquire, release, size, rs, rt, rn } => {
                (size.size_field() << 30)
                    | 0x08A0_7C00
                    | ((acquire as u32) << 22)
                    | ((release as u32) << 15)
                    | (rs.encoding() << 16)
                    | (rn.encoding() << 5)
                    | rt.encoding()
            }
            Inst::Dmb { option } => {
                let crm: u32 = match option {
                    DmbOption::Ish => 0b1011,
                    DmbOption::IshLd => 0b1001,
                    DmbOption::IshSt => 0b1010,
                };
                0xD503_30BF | (crm << 8)
            }
        }
    }
}

fn encode_add_sub_imm(
    op: AddSub,
    size: OperandSize,
    set_flags: bool,
    rd: Gpr,
    rn: Gpr,
    imm12: u16,
    shift12: bool,
) -> u32 {
    let op_bit = match op {
        AddSub::Add => 0,
        AddSub::Sub => 1,
    };
    debug_assert!(imm12 < (1 << 12));
    (size.sf() << 31)
        | (op_bit << 30)
        | ((set_flags as u32) << 29)
        | (0b100010 << 23)
        | ((shift12 as u32) << 22)
        | ((imm12 as u32) << 10)
        | (rn.encoding() << 5)
        | rd.encoding()
}

fn encode_logical(op: LogicOp, size: OperandSize, rd: Gpr, rn: Gpr, rm: Gpr, amount: u8) -> u32 {
    let opc: u32 = match op {
        LogicOp::And => 0b00,
        LogicOp::Orr => 0b01,
        LogicOp::Eor => 0b10,
        LogicOp::Ands => 0b11,
    };
    (size.sf() << 31)
        | (opc << 29)
        | (0b01010 << 24)
        | (rm.encoding() << 16)
        | ((amount as u32) << 10)
        | (rn.encoding() << 5)
        | rd.encoding()
}

/// Patch a 26-bit branch displacement (in instructions, signed) into a `b`/`bl` word.
pub fn patch_imm26(word: u32, insns: i32) -> u32 {
    (word & !0x03FF_FFFF) | ((insns as u32) & 0x03FF_FFFF)
}

/// Patch a 19-bit displacement (in instructions, signed) into a `b.cond`/`cbz` word.
pub fn patch_imm19(word: u32, insns: i32) -> u32 {
    (word & !(0x7FFFF << 5)) | (((insns as u32) & 0x7FFFF) << 5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mach::reg::*;

    #[test]
    fn known_encodings() {
        // ret
        assert_eq!(Inst::Ret { rn: LR }.encode(), 0xD65F03C0);
        // nop
        assert_eq!(Inst::Nop.encode(), 0xD503201F);
        // movz x0, #0
        assert_eq!(
            Inst::MovWide { kind: MovKind::Zero, size: OperandSize::S64, rd: X0, imm16: 0, shift: 0 }
                .encode(),
            0xD2800000
        );
        // movz w0, #1
        assert_eq!(
            Inst::MovWide { kind: MovKind::Zero, size: OperandSize::S32, rd: X0, imm16: 1, shift: 0 }
                .encode(),
            0x52800020
        );
        // add x0, x0, x1
        assert_eq!(
            Inst::AddSubReg {
                op: AddSub::Add,
                size: OperandSize::S64,
                set_flags: false,
                rd: X0,
                rn: X0,
                rm: X1,
                amount: 0,
            }
            .encode(),
            0x8B010000
        );
        // sub sp, sp, #16
        assert_eq!(
            Inst::AddSubImm {
                op: AddSub::Sub,
                size: OperandSize::S64,
                set_flags: false,
                rd: SP,
                rn: SP,
                imm12: 16,
                shift12: false,
            }
            .encode(),
            0xD10043FF
        );
        // orr x0, xzr, x1  (mov x0, x1)
        assert_eq!(Inst::MovReg { size: OperandSize::S64, rd: X0, rm: X1 }.encode(), 0xAA0103E0);
        // mul x0, x1, x2  == madd x0, x1, x2, xzr
        assert_eq!(
            Inst::Madd { size: OperandSize::S64, rd: X0, rn: X1, rm: X2, ra: ZR }.encode(),
            0x9B027C20
        );
        // udiv x0, x1, x2
        assert_eq!(
            Inst::DataProc2 {
                op: DataProc2::Udiv,
                size: OperandSize::S64,
                rd: X0,
                rn: X1,
                rm: X2,
            }
            .encode(),
            0x9AC20820
        );
        // clz x0, x1
        assert_eq!(
            Inst::DataProc1 { op: DataProc1::Clz, size: OperandSize::S64, rd: X0, rn: X1 }.encode(),
            0xDAC01020
        );
        // clz w0, w1
        assert_eq!(
            Inst::DataProc1 { op: DataProc1::Clz, size: OperandSize::S32, rd: X0, rn: X1 }.encode(),
            0x5AC01020
        );
        // rbit x0, x1
        assert_eq!(
            Inst::DataProc1 { op: DataProc1::Rbit, size: OperandSize::S64, rd: X0, rn: X1 }.encode(),
            0xDAC00020
        );
        // rev w0, w1
        assert_eq!(
            Inst::DataProc1 { op: DataProc1::Rev, size: OperandSize::S32, rd: X0, rn: X1 }.encode(),
            0x5AC00820
        );
        // rev x0, x1
        assert_eq!(
            Inst::DataProc1 { op: DataProc1::Rev, size: OperandSize::S64, rd: X0, rn: X1 }.encode(),
            0xDAC00C20
        );
        // sxtb w0, w1
        assert_eq!(
            Inst::Sxt { from: MemSize::B, to: OperandSize::S32, rd: X0, rn: X1 }.encode(),
            0x13001C20
        );
        // sxth w0, w1
        assert_eq!(
            Inst::Sxt { from: MemSize::H, to: OperandSize::S32, rd: X0, rn: X1 }.encode(),
            0x13003C20
        );
        // sxtb x0, w1
        assert_eq!(
            Inst::Sxt { from: MemSize::B, to: OperandSize::S64, rd: X0, rn: X1 }.encode(),
            0x93401C20
        );
        // sxtw x0, w1
        assert_eq!(
            Inst::Sxt { from: MemSize::W, to: OperandSize::S64, rd: X0, rn: X1 }.encode(),
            0x93407C20
        );
        // csel x0, x1, x2, eq
        assert_eq!(
            Inst::CondSel {
                op: CondSel::Csel,
                size: OperandSize::S64,
                rd: X0,
                rn: X1,
                rm: X2,
                cond: Cond::Eq,
            }
            .encode(),
            0x9A820020
        );
        // stp x29, x30, [sp, #-16]!
        assert_eq!(
            Inst::LoadStorePair {
                load: false,
                size: OperandSize::S64,
                index: PairIndex::PreIndex,
                rt: FP,
                rt2: LR,
                rn: SP,
                offset: -16,
            }
            .encode(),
            0xA9BF7BFD
        );
        // ldp x29, x30, [sp], #16
        assert_eq!(
            Inst::LoadStorePair {
                load: true,
                size: OperandSize::S64,
                index: PairIndex::PostIndex,
                rt: FP,
                rt2: LR,
                rn: SP,
                offset: 16,
            }
            .encode(),
            0xA8C17BFD
        );
        // str x0, [sp, #8]
        assert_eq!(
            Inst::LoadStoreUImm {
                load: false,
                signed: false,
                size: MemSize::X,
                rt: X0,
                rn: SP,
                offset: 8,
            }
            .encode(),
            0xF90007E0
        );
        // ldr x0, [sp, #8]
        assert_eq!(
            Inst::LoadStoreUImm {
                load: true,
                signed: false,
                size: MemSize::X,
                rt: X0,
                rn: SP,
                offset: 8,
            }
            .encode(),
            0xF94007E0
        );
    }

    #[test]
    fn fp_encodings() {
        // fadd d16, d16, d17
        assert_eq!(
            Inst::FpDataProc2 { op: FpOp2::Fadd, size: FpSize::S64, rd: V16, rn: V16, rm: V17 }
                .encode(),
            0x1E712A10
        );
        // fsub d16, d16, d17
        assert_eq!(
            Inst::FpDataProc2 { op: FpOp2::Fsub, size: FpSize::S64, rd: V16, rn: V16, rm: V17 }
                .encode(),
            0x1E713A10
        );
        // fmul d16, d16, d17
        assert_eq!(
            Inst::FpDataProc2 { op: FpOp2::Fmul, size: FpSize::S64, rd: V16, rn: V16, rm: V17 }
                .encode(),
            0x1E710A10
        );
        // fdiv d16, d16, d17
        assert_eq!(
            Inst::FpDataProc2 { op: FpOp2::Fdiv, size: FpSize::S64, rd: V16, rn: V16, rm: V17 }
                .encode(),
            0x1E711A10
        );
        // fadd s16, s16, s17
        assert_eq!(
            Inst::FpDataProc2 { op: FpOp2::Fadd, size: FpSize::S32, rd: V16, rn: V16, rm: V17 }
                .encode(),
            0x1E312A10
        );
        // fneg d16, d16
        assert_eq!(
            Inst::FpDataProc1 { op: FpOp1::Fneg, size: FpSize::S64, rd: V16, rn: V16 }.encode(),
            0x1E614210
        );
        // fneg s16, s16
        assert_eq!(
            Inst::FpDataProc1 { op: FpOp1::Fneg, size: FpSize::S32, rd: V16, rn: V16 }.encode(),
            0x1E214210
        );
        // fmov d0, x9
        assert_eq!(Inst::FmovFromGpr { size: FpSize::S64, rd: V0, rn: X9 }.encode(), 0x9E670120);
        // fmov s0, w9
        assert_eq!(Inst::FmovFromGpr { size: FpSize::S32, rd: V0, rn: X9 }.encode(), 0x1E270120);
        // str d0, [sp, #8]
        assert_eq!(
            Inst::LoadStoreFpUImm { load: false, size: FpSize::S64, rt: V0, rn: SP, offset: 8 }
                .encode(),
            0xFD0007E0
        );
        // ldr d0, [sp, #8]
        assert_eq!(
            Inst::LoadStoreFpUImm { load: true, size: FpSize::S64, rt: V0, rn: SP, offset: 8 }
                .encode(),
            0xFD4007E0
        );
        // str s0, [sp, #4]
        assert_eq!(
            Inst::LoadStoreFpUImm { load: false, size: FpSize::S32, rt: V0, rn: SP, offset: 4 }
                .encode(),
            0xBD0007E0
        );
        // fcmp d0, d1
        assert_eq!(Inst::FpCmp { size: FpSize::S64, rn: V0, rm: V1 }.encode(), 0x1E612000);
        // scvtf d0, x9
        assert_eq!(
            Inst::IntToFp { signed: true, fp: FpSize::S64, int: OperandSize::S64, rd: V0, rn: X9 }
                .encode(),
            0x9E620120
        );
        // ucvtf d0, x9
        assert_eq!(
            Inst::IntToFp { signed: false, fp: FpSize::S64, int: OperandSize::S64, rd: V0, rn: X9 }
                .encode(),
            0x9E630120
        );
        // scvtf s0, w9
        assert_eq!(
            Inst::IntToFp { signed: true, fp: FpSize::S32, int: OperandSize::S32, rd: V0, rn: X9 }
                .encode(),
            0x1E220120
        );
        // fcvtzs x9, d0
        assert_eq!(
            Inst::FpToInt { signed: true, fp: FpSize::S64, int: OperandSize::S64, rd: X9, rn: V0 }
                .encode(),
            0x9E780009
        );
        // fcvtzu x9, d0
        assert_eq!(
            Inst::FpToInt { signed: false, fp: FpSize::S64, int: OperandSize::S64, rd: X9, rn: V0 }
                .encode(),
            0x9E790009
        );
        // fcvtzs w9, s0
        assert_eq!(
            Inst::FpToInt { signed: true, fp: FpSize::S32, int: OperandSize::S32, rd: X9, rn: V0 }
                .encode(),
            0x1E380009
        );
        // fcvt s0, d0
        assert_eq!(
            Inst::FpCvt { from: FpSize::S64, to: FpSize::S32, rd: V0, rn: V0 }.encode(),
            0x1E624000
        );
        // fcvt d0, s0
        assert_eq!(
            Inst::FpCvt { from: FpSize::S32, to: FpSize::S64, rd: V0, rn: V0 }.encode(),
            0x1E22C000
        );
        // umulh x9, x10, x11
        assert_eq!(
            Inst::MulHigh { signed: false, rd: X9, rn: X10, rm: X11 }.encode(),
            0x9BCB7D49
        );
        // smulh x9, x10, x11
        assert_eq!(
            Inst::MulHigh { signed: true, rd: X9, rn: X10, rm: X11 }.encode(),
            0x9B4B7D49
        );
    }

    #[test]
    fn atomic_encodings() {
        // ldar x0, [x1] / ldarb w0, [x1]
        assert_eq!(Inst::LoadAcq { size: MemSize::X, rt: X0, rn: X1 }.encode(), 0xC8DFFC20);
        assert_eq!(Inst::LoadAcq { size: MemSize::B, rt: X0, rn: X1 }.encode(), 0x08DFFC20);
        // stlr x0, [x1] / stlrh w0, [x1]
        assert_eq!(Inst::StoreRel { size: MemSize::X, rt: X0, rn: X1 }.encode(), 0xC89FFC20);
        assert_eq!(Inst::StoreRel { size: MemSize::H, rt: X0, rn: X1 }.encode(), 0x489FFC20);
        // ldaddal x2, x0, [x1]  (rs=x2, rt=x0, rn=x1)
        assert_eq!(
            Inst::AtomicRmw {
                op: AtomicRmwOp::Add,
                acquire: true,
                release: true,
                size: MemSize::X,
                rs: X2,
                rt: X0,
                rn: X1,
            }
            .encode(),
            0xF8E20020
        );
        // ldadd x2, x0, [x1] (relaxed)
        assert_eq!(
            Inst::AtomicRmw {
                op: AtomicRmwOp::Add,
                acquire: false,
                release: false,
                size: MemSize::X,
                rs: X2,
                rt: X0,
                rn: X1,
            }
            .encode(),
            0xF8220020
        );
        // ldsetal x2, x0, [x1]
        assert_eq!(
            Inst::AtomicRmw {
                op: AtomicRmwOp::Set,
                acquire: true,
                release: true,
                size: MemSize::X,
                rs: X2,
                rt: X0,
                rn: X1,
            }
            .encode(),
            0xF8E23020
        );
        // swpal x2, x0, [x1]
        assert_eq!(
            Inst::AtomicRmw {
                op: AtomicRmwOp::Swp,
                acquire: true,
                release: true,
                size: MemSize::X,
                rs: X2,
                rt: X0,
                rn: X1,
            }
            .encode(),
            0xF8E28020
        );
        // ldaddalb w2, w0, [x1]
        assert_eq!(
            Inst::AtomicRmw {
                op: AtomicRmwOp::Add,
                acquire: true,
                release: true,
                size: MemSize::B,
                rs: X2,
                rt: X0,
                rn: X1,
            }
            .encode(),
            0x38E20020
        );
        // casal x0, x2, [x1]  (rs=x0, rt=x2, rn=x1)
        assert_eq!(
            Inst::AtomicCas {
                acquire: true,
                release: true,
                size: MemSize::X,
                rs: X0,
                rt: X2,
                rn: X1,
            }
            .encode(),
            0xC8E0FC22
        );
        // casalb w0, w2, [x1]
        assert_eq!(
            Inst::AtomicCas {
                acquire: true,
                release: true,
                size: MemSize::B,
                rs: X0,
                rt: X2,
                rn: X1,
            }
            .encode(),
            0x08E0FC22
        );
        // dmb ish / ishld / ishst
        assert_eq!(Inst::Dmb { option: DmbOption::Ish }.encode(), 0xD5033BBF);
        assert_eq!(Inst::Dmb { option: DmbOption::IshLd }.encode(), 0xD50339BF);
        assert_eq!(Inst::Dmb { option: DmbOption::IshSt }.encode(), 0xD5033ABF);
    }
}
