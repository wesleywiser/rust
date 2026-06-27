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
    /// `udiv`/`sdiv`/`lslv`/`lsrv`/`asrv rd, rn, rm`.
    DataProc2 { op: DataProc2, size: OperandSize, rd: Gpr, rn: Gpr, rm: Gpr },

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

    /// `fmov`/`fadd`/`fsub`/`fmul`/`fdiv` two-operand FP (placeholder for the FP batch).
    FpDataProc2 { op: FpOp2, size: FpSize, rd: Vreg, rn: Vreg, rm: Vreg },
}

/// Two-operand floating-point opcode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FpOp2 {
    Fadd,
    Fsub,
    Fmul,
    Fdiv,
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
}
