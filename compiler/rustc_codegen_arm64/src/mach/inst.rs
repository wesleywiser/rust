//! The AArch64 instruction model and its binary (`u32`) encoder.
//!
//! Each [`Inst`] lowers to either a 32-bit machine word (via [`Inst::encode`], used for object
//! emission) or textual assembly (see `emit_asm`). Instructions that reference labels or symbols
//! encode with zeroed immediate fields; the containing function patches branch displacements and
//! records relocations during layout.

use crate::mach::reg::{Cond, FpSize, Gpr, OperandSize, Vreg};

/// Index of a local label within a function.
pub type Label = u32;

/// A resolved source location for debug info, attached to the instruction stream via the
/// [`Inst::DebugLoc`] pseudo-op. The `file`/`line`/`col` fields are already resolved against the
/// source map (so the object-emitting back-half needs no `TyCtxt`): `file` is an index into the
/// debug context's file table, `line`/`col` are 1-based (0 meaning "unknown"). They describe the
/// *innermost* (most-inlined) source location, which is what the `.debug_line` program records.
///
/// `loc` indexes the debug context's location table, which additionally records the lexical scope
/// and the inlined-at call chain; it is used to reconstruct `DW_TAG_inlined_subroutine` frames.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DebugLoc {
    pub file: u32,
    pub line: u32,
    pub col: u32,
    pub loc: u32,
}

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
    /// `crc32{b,h,w,x}` — one CRC32 step (the zlib/PNG polynomial). The suffix is the data width;
    /// the accumulator (`rd`/`rn`) is always 32-bit, so the operand `size` only selects the `rm`
    /// data-operand width (32-bit for `b`/`h`/`w`, 64-bit for `x`).
    Crc32b,
    Crc32h,
    Crc32w,
    Crc32x,
    /// `crc32c{b,h,w,x}` — the CRC32C (Castagnoli) polynomial variants.
    Crc32cb,
    Crc32ch,
    Crc32cw,
    Crc32cx,
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

    /// A source-location marker for debug info; emits no bytes. Layout records its byte offset so
    /// the line-number program can map code addresses back to `(file, line, col)`.
    DebugLoc(DebugLoc),

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
    /// `adc`/`adcs`/`sbc`/`sbcs rd, rn, rm` — add/subtract with carry, propagating the carry flag.
    /// Used to chain 64-bit operations into 128-bit arithmetic (the high word consumes the carry the
    /// low word produced). `negs`/`ngc` are the `rn == xzr` forms. With `set_flags`, the resulting
    /// `NZCV` reflects the full 128-bit result for overflow detection.
    AddSubCarry { op: AddSub, size: OperandSize, set_flags: bool, rd: Gpr, rn: Gpr, rm: Gpr },
    /// `add`/`sub rd, rn, rm` (extended-register form, `UXTX #0`). Unlike the shifted-register
    /// form, this permits the stack pointer as `rd`/`rn`, so it is used to adjust `sp` by, or form
    /// frame addresses from, an offset materialized in a register (for frames too large for an
    /// immediate).
    AddSubExtReg { op: AddSub, size: OperandSize, rd: Gpr, rn: Gpr, rm: Gpr },
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

    /// Load/store with an unsigned, scaled 12-bit immediate offset: `[rn, #imm]`. The offset is a
    /// `u64` so it can address arbitrarily large frames; the builder routes offsets that exceed the
    /// encodable range through a register instead (see `builder::push_mem_*`).
    LoadStoreUImm { load: bool, signed: bool, size: MemSize, rt: Gpr, rn: Gpr, offset: u64 },
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
    /// `adrp rd, <sym>@TLVPPAGE` — thread-local descriptor page (records a `TlvpPage21` relocation).
    AdrpTlv { rd: Gpr, sym: SymRef },
    /// `ldr rt, [rn, <sym>@TLVPPAGEOFF]` — load the descriptor address (records a `TlvpPageOff12`
    /// relocation that patches the scaled 12-bit immediate).
    LdrTlvLo { rt: Gpr, rn: Gpr, sym: SymRef },
    /// `adrp rd, <sym>@GOTPAGE` — GOT entry page (records a `GotLoadPage21` relocation). Used for
    /// dylib-imported statics (e.g. `_mach_task_self_`) whose address is only known via the GOT.
    AdrpGot { rd: Gpr, sym: SymRef },
    /// `ldr rt, [rn, <sym>@GOTPAGEOFF]` — load the GOT slot (records a `GotLoadPageOff12`).
    LdrGotLo { rt: Gpr, rn: Gpr, sym: SymRef },

    /// `fadd`/`fsub`/`fmul`/`fdiv` two-operand FP.
    FpDataProc2 { op: FpOp2, size: FpSize, rd: Vreg, rn: Vreg, rm: Vreg },
    /// `fneg`/`fabs`/`fsqrt`/`frint{n,p,m,z,a}` one-operand FP.
    FpDataProc1 { op: FpOp1, size: FpSize, rd: Vreg, rn: Vreg },
    /// `fmadd rd, rn, rm, ra` — fused multiply-add (`rd = rn * rm + ra`, single rounding).
    FpFma { size: FpSize, rd: Vreg, rn: Vreg, rm: Vreg, ra: Vreg },
    /// `fmov rd, rn` moving a general register's raw bits into a scalar FP register (used to
    /// materialize floating-point constants).
    FmovFromGpr { size: FpSize, rd: Vreg, rn: Gpr },
    /// `ldr`/`str` of a scalar FP register with an unsigned, scaled 12-bit immediate offset.
    LoadStoreFpUImm { load: bool, size: FpSize, rt: Vreg, rn: Gpr, offset: u64 },
    /// `fcmp rn, rm` — floating-point compare, setting the NZCV flags.
    FpCmp { size: FpSize, rn: Vreg, rm: Vreg },
    /// `scvtf`/`ucvtf` — integer (in a GPR) to floating-point conversion.
    IntToFp { signed: bool, fp: FpSize, int: OperandSize, rd: Vreg, rn: Gpr },
    /// `fcvtzs`/`fcvtzu` — floating-point to integer, rounding toward zero (saturating in hardware).
    FpToInt { signed: bool, fp: FpSize, int: OperandSize, rd: Gpr, rn: Vreg },
    /// `fcvt` — floating-point precision conversion (`f32`<->`f64`).
    FpCvt { from: FpSize, to: FpSize, rd: Vreg, rn: Vreg },

    /// `ldr`/`str` of a full 128-bit vector (`q`) register with an unsigned, scaled 12-bit offset.
    /// Used to move whole vectors between frame slots and `v` registers (e.g. for crypto).
    LoadStoreQ { load: bool, rt: Vreg, rn: Gpr, offset: u64 },
    /// Two-register ARMv8 crypto instruction (`aese`/`aesd`/`aesmc`/`aesimc`, `sha256su0`, `sha1h`,
    /// `sha1su1`). `rd` is read-modify-write for the AES/`su0`/`su1` forms.
    CryptoTwo { op: CryptoTwoOp, rd: Vreg, rn: Vreg },
    /// Three-register ARMv8 crypto instruction (`sha256h`/`sha256h2`/`sha256su1`, `sha1c`/`sha1p`/
    /// `sha1m`/`sha1su0`). `rd` is read-modify-write.
    CryptoThree { op: CryptoThreeOp, rd: Vreg, rn: Vreg, rm: Vreg },
    /// Three-register NEON "3-same" vector instruction (`add`/`sub`/`mul`, `and`/`orr`/`eor`,
    /// `fadd`/`fsub`/`fmul`/`fdiv`). `q` selects a 128-bit (`true`) vs 64-bit (`false`) vector; the
    /// 2-bit `size` field (bits 23:22) carries the integer lane size (`00`/`01`/`10`/`11` =
    /// 8/16/32/64-bit) or, for the float ops, the single `sz` bit (`0`=f32, `1`=f64). Bitwise ops
    /// fix `size` in their base, so they pass `0`.
    SimdThree { op: SimdOp, q: bool, size: u8, rd: Vreg, rn: Vreg, rm: Vreg },
    /// Two-register NEON "2-misc" vector instruction (`neg`/`not`, `fneg`/`fabs`/`fsqrt`, the
    /// `frint*` rounding modes). `q`/`size` are interpreted as for [`Inst::SimdThree`].
    SimdTwo { op: SimdUnOp, q: bool, size: u8, rd: Vreg, rn: Vreg },

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
    /// `isb sy` — instruction synchronization barrier (used by `spin_loop` hints).
    Isb,
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

/// Two-register ARMv8 crypto-extension opcode (`rd`/`rn`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CryptoTwoOp {
    Aese,
    Aesd,
    Aesmc,
    Aesimc,
    Sha256su0,
    Sha1h,
    Sha1su1,
    Sha512su0,
}

/// Three-register ARMv8 crypto-extension opcode (`rd`/`rn`/`rm`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CryptoThreeOp {
    Sha256h,
    Sha256h2,
    Sha256su1,
    Sha1c,
    Sha1p,
    Sha1m,
    Sha1su0,
    Sha512h,
    Sha512h2,
    Sha512su1,
}

/// NEON "3-same" three-register vector opcode (see [`Inst::SimdThree`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SimdOp {
    Add,
    Sub,
    Mul,
    And,
    Orr,
    Eor,
    Fadd,
    Fsub,
    Fmul,
    Fdiv,
    /// Lane-wise compares producing an all-ones/all-zero mask (`cmeq`/`cmgt`/`cmge` signed,
    /// `cmhi`/`cmhs` unsigned; `fcmeq`/`fcmgt`/`fcmge` float).
    Cmeq,
    Cmgt,
    Cmge,
    Cmhi,
    Cmhs,
    Fcmeq,
    Fcmgt,
    Fcmge,
}

/// NEON "2-misc" two-register vector opcode (see [`Inst::SimdTwo`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SimdUnOp {
    /// Integer negate (`neg`) and bitwise not (`not`/`mvn`).
    Neg,
    Not,
    /// Floating-point negate / absolute / square-root.
    Fneg,
    Fabs,
    Fsqrt,
    /// Floating-point round-to-integral (the `frint*` rounding modes).
    Frintn,
    Frintp,
    Frintm,
    Frintz,
    Frinta,
}

/// One-operand floating-point opcode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FpOp1 {
    Fabs,
    Fneg,
    Fsqrt,
    /// Round to nearest, ties to even (`frintn`).
    Frintn,
    /// Round toward +∞ (`frintp`) — `ceil`.
    Frintp,
    /// Round toward -∞ (`frintm`) — `floor`.
    Frintm,
    /// Round toward zero (`frintz`) — `trunc`.
    Frintz,
    /// Round to nearest, ties away from zero (`frinta`) — `round`.
    Frinta,
}

impl Inst {
    /// Encode to a 32-bit little-endian instruction word.
    ///
    /// Label-relative branches and symbol references encode with zeroed immediate fields; the
    /// containing function patches branch displacements and emits relocations during layout.
    pub fn encode(&self) -> u32 {
        match *self {
            Inst::Label(_) => panic!("Inst::Label has no encoding; resolve labels during layout"),

            Inst::DebugLoc(_) => {
                panic!("Inst::DebugLoc has no encoding; it is stripped during layout")
            }

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
                // The shift amount (LSL) occupies a 6-bit field and must be < the register width.
                debug_assert!(amount < if matches!(size, OperandSize::S64) { 64 } else { 32 });
                (size.sf() << 31)
                    | (op_bit << 30)
                    | ((set_flags as u32) << 29)
                    | (0b01011 << 24)
                    | (rm.encoding() << 16)
                    | ((amount as u32) << 10)
                    | (rn.encoding() << 5)
                    | rd.encoding()
            }
            Inst::AddSubCarry { op, size, set_flags, rd, rn, rm } => {
                let op_bit = match op {
                    AddSub::Add => 0,
                    AddSub::Sub => 1,
                };
                (size.sf() << 31)
                    | (op_bit << 30)
                    | ((set_flags as u32) << 29)
                    | (0b11010000 << 21)
                    | (rm.encoding() << 16)
                    | (rn.encoding() << 5)
                    | rd.encoding()
            }
            Inst::AddSubExtReg { op, size, rd, rn, rm } => {
                let op_bit = match op {
                    AddSub::Add => 0,
                    AddSub::Sub => 1,
                };
                // Extended-register form with `UXTX #0` (option=0b011, imm3=0): a plain 64-bit
                // register operand that, unlike the shifted form, allows `sp` as `rd`/`rn`.
                (size.sf() << 31)
                    | (op_bit << 30)
                    | (0b01011 << 24)
                    | (1 << 21)
                    | (rm.encoding() << 16)
                    | (0b011 << 13)
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
                    DataProc2::Crc32b => 0b010000,
                    DataProc2::Crc32h => 0b010001,
                    DataProc2::Crc32w => 0b010010,
                    DataProc2::Crc32x => 0b010011,
                    DataProc2::Crc32cb => 0b010100,
                    DataProc2::Crc32ch => 0b010101,
                    DataProc2::Crc32cw => 0b010110,
                    DataProc2::Crc32cx => 0b010111,
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
                let bytes = size.bytes() as u64;
                debug_assert!(offset % bytes == 0, "unaligned scaled offset");
                let scaled = offset / bytes;
                debug_assert!(scaled < (1 << 12), "offset out of range for unsigned-imm form");
                (size.size_field() << 30)
                    | (0b111 << 27)
                    | (0b01 << 24)
                    | (opc << 22)
                    | ((scaled as u32) << 10)
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
            // `adrp` page of a thread-local descriptor; encodes like `adrp`, the relocation differs.
            Inst::AdrpTlv { rd, .. } => 0x90000000 | rd.encoding(),
            // `ldr rt, [rn, #0]` (64-bit); the `TlvpPageOff12` relocation patches the immediate.
            Inst::LdrTlvLo { rt, rn, .. } => 0xF940_0000 | (rn.encoding() << 5) | rt.encoding(),
            // GOT page/offset; encode like adrp + 64-bit ldr, the GOT relocations patch them.
            Inst::AdrpGot { rd, .. } => 0x90000000 | rd.encoding(),
            Inst::LdrGotLo { rt, rn, .. } => 0xF940_0000 | (rn.encoding() << 5) | rt.encoding(),
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
                    FpOp1::Frintn => 0b001000,
                    FpOp1::Frintp => 0b001001,
                    FpOp1::Frintm => 0b001010,
                    FpOp1::Frintz => 0b001011,
                    FpOp1::Frinta => 0b001100,
                };
                (0b00011110 << 24)
                    | (size.ftype() << 22)
                    | (1 << 21)
                    | (opcode << 15)
                    | (0b10000 << 10)
                    | (rn.encoding() << 5)
                    | rd.encoding()
            }
            // Floating-point data-processing (3 source); FMADD has o1=0, o0=0.
            Inst::FpFma { size, rd, rn, rm, ra } => {
                (0b00011111 << 24)
                    | (size.ftype() << 22)
                    | (rm.encoding() << 16)
                    | (ra.encoding() << 10)
                    | (rn.encoding() << 5)
                    | rd.encoding()
            }
            Inst::FmovFromGpr { size, rd, rn } => {
                let base: u32 = match size {
                    FpSize::S64 => 0x9E67_0000,
                    FpSize::S32 => 0x1E27_0000,
                    FpSize::S16 => 0x1EE7_0000,
                };
                base | (rn.encoding() << 5) | rd.encoding()
            }
            Inst::LoadStoreFpUImm { load, size, rt, rn, offset } => {
                let base: u32 = match (load, size) {
                    (false, FpSize::S64) => 0xFD00_0000,
                    (true, FpSize::S64) => 0xFD40_0000,
                    (false, FpSize::S32) => 0xBD00_0000,
                    (true, FpSize::S32) => 0xBD40_0000,
                    (false, FpSize::S16) => 0x7D00_0000,
                    (true, FpSize::S16) => 0x7D40_0000,
                };
                let scale = match size {
                    FpSize::S16 => 2,
                    FpSize::S32 => 4,
                    FpSize::S64 => 8,
                };
                debug_assert!(offset % scale == 0, "unaligned scaled FP offset");
                let scaled = offset / scale;
                debug_assert!(scaled < (1 << 12), "FP offset out of range for unsigned-imm form");
                base | ((scaled as u32) << 10) | (rn.encoding() << 5) | rt.encoding()
            }
            Inst::FpCmp { size, rn, rm } => {
                let base: u32 = match size {
                    FpSize::S64 => 0x1E60_2000,
                    FpSize::S32 => 0x1E20_2000,
                    FpSize::S16 => 0x1EE0_2000,
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
                    (FpSize::S16, FpSize::S32) => 0x1EE2_4000,
                    (FpSize::S32, FpSize::S16) => 0x1E23_C000,
                    (FpSize::S16, FpSize::S64) => 0x1EE2_C000,
                    (FpSize::S64, FpSize::S16) => 0x1E63_C000,
                    // Same-size "cast" is an identity copy: emit `fmov`, not a real convert.
                    (FpSize::S32, FpSize::S32) => 0x1E20_4000,
                    (FpSize::S64, FpSize::S64) => 0x1E60_4000,
                    (FpSize::S16, FpSize::S16) => 0x1EE0_4000,
                };
                base | (rn.encoding() << 5) | rd.encoding()
            }

            Inst::LoadStoreQ { load, rt, rn, offset } => {
                let base: u32 = if load { 0x3DC0_0000 } else { 0x3D80_0000 };
                debug_assert!(offset % 16 == 0, "unaligned scaled 128-bit FP offset");
                let scaled = offset / 16;
                debug_assert!(scaled < (1 << 12), "q offset out of range for unsigned-imm form");
                base | ((scaled as u32) << 10) | (rn.encoding() << 5) | rt.encoding()
            }
            Inst::CryptoTwo { op, rd, rn } => {
                let base: u32 = match op {
                    CryptoTwoOp::Aese => 0x4E28_4800,
                    CryptoTwoOp::Aesd => 0x4E28_5800,
                    CryptoTwoOp::Aesmc => 0x4E28_6800,
                    CryptoTwoOp::Aesimc => 0x4E28_7800,
                    CryptoTwoOp::Sha256su0 => 0x5E28_2800,
                    CryptoTwoOp::Sha1h => 0x5E28_0800,
                    CryptoTwoOp::Sha1su1 => 0x5E28_1800,
                    CryptoTwoOp::Sha512su0 => 0xCEC0_8000,
                };
                base | (rn.encoding() << 5) | rd.encoding()
            }
            Inst::CryptoThree { op, rd, rn, rm } => {
                let base: u32 = match op {
                    CryptoThreeOp::Sha256h => 0x5E00_4000,
                    CryptoThreeOp::Sha256h2 => 0x5E00_5000,
                    CryptoThreeOp::Sha256su1 => 0x5E00_6000,
                    CryptoThreeOp::Sha1c => 0x5E00_0000,
                    CryptoThreeOp::Sha1p => 0x5E00_1000,
                    CryptoThreeOp::Sha1m => 0x5E00_2000,
                    CryptoThreeOp::Sha1su0 => 0x5E00_3000,
                    CryptoThreeOp::Sha512h => 0xCE60_8000,
                    CryptoThreeOp::Sha512h2 => 0xCE60_8400,
                    CryptoThreeOp::Sha512su1 => 0xCE60_8800,
                };
                base | (rm.encoding() << 16) | (rn.encoding() << 5) | rd.encoding()
            }
            Inst::SimdThree { op, q, size, rd, rn, rm } => {
                let base: u32 = match op {
                    SimdOp::Add => 0x0E20_8400,
                    SimdOp::Sub => 0x2E20_8400,
                    SimdOp::Mul => 0x0E20_9C00,
                    SimdOp::And => 0x0E20_1C00,
                    SimdOp::Orr => 0x0EA0_1C00,
                    SimdOp::Eor => 0x2E20_1C00,
                    SimdOp::Fadd => 0x0E20_D400,
                    SimdOp::Fsub => 0x0EA0_D400,
                    SimdOp::Fmul => 0x2E20_DC00,
                    SimdOp::Fdiv => 0x2E20_FC00,
                    SimdOp::Cmeq => 0x2E20_8C00,
                    SimdOp::Cmgt => 0x0E20_3400,
                    SimdOp::Cmge => 0x0E20_3C00,
                    SimdOp::Cmhi => 0x2E20_3400,
                    SimdOp::Cmhs => 0x2E20_3C00,
                    SimdOp::Fcmeq => 0x0E20_E400,
                    SimdOp::Fcmgt => 0x2EA0_E400,
                    SimdOp::Fcmge => 0x2E20_E400,
                };
                debug_assert!(size < 4, "NEON size/sz field is 2 bits");
                base | ((q as u32) << 30)
                    | ((size as u32) << 22)
                    | (rm.encoding() << 16)
                    | (rn.encoding() << 5)
                    | rd.encoding()
            }
            Inst::SimdTwo { op, q, size, rd, rn } => {
                let base: u32 = match op {
                    SimdUnOp::Neg => 0x2E20_B800,
                    SimdUnOp::Not => 0x2E20_5800,
                    SimdUnOp::Fneg => 0x2EA0_F800,
                    SimdUnOp::Fabs => 0x0EA0_F800,
                    SimdUnOp::Fsqrt => 0x2EA1_F800,
                    SimdUnOp::Frintn => 0x0E21_8800,
                    SimdUnOp::Frintp => 0x0EA1_8800,
                    SimdUnOp::Frintm => 0x0E21_9800,
                    SimdUnOp::Frintz => 0x0EA1_9800,
                    SimdUnOp::Frinta => 0x2E21_8800,
                };
                debug_assert!(size < 4, "NEON size/sz field is 2 bits");
                base | ((q as u32) << 30)
                    | ((size as u32) << 22)
                    | (rn.encoding() << 5)
                    | rd.encoding()
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
            Inst::Isb => 0xD503_3FDF,
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
        // crc32b w0, w1, w2  (32-bit data operand -> sf=0)
        assert_eq!(
            Inst::DataProc2 { op: DataProc2::Crc32b, size: OperandSize::S32, rd: X0, rn: X1, rm: X2 }
                .encode(),
            0x1AC24020
        );
        // crc32x w0, w1, x2  (64-bit data operand -> sf=1)
        assert_eq!(
            Inst::DataProc2 { op: DataProc2::Crc32x, size: OperandSize::S64, rd: X0, rn: X1, rm: X2 }
                .encode(),
            0x9AC24C20
        );
        // crc32cb w0, w1, w2  (Castagnoli)
        assert_eq!(
            Inst::DataProc2 { op: DataProc2::Crc32cb, size: OperandSize::S32, rd: X0, rn: X1, rm: X2 }
                .encode(),
            0x1AC25020
        );
        // crc32cx w0, w1, x2  (Castagnoli, 64-bit data)
        assert_eq!(
            Inst::DataProc2 { op: DataProc2::Crc32cx, size: OperandSize::S64, rd: X0, rn: X1, rm: X2 }
                .encode(),
            0x9AC25C20
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
        // fabs d0, d0 / fsqrt d0, d0
        assert_eq!(
            Inst::FpDataProc1 { op: FpOp1::Fabs, size: FpSize::S64, rd: V0, rn: V0 }.encode(),
            0x1E60C000
        );
        assert_eq!(
            Inst::FpDataProc1 { op: FpOp1::Fsqrt, size: FpSize::S64, rd: V0, rn: V0 }.encode(),
            0x1E61C000
        );
        // frint{m,p,z,a,n} d0, d0 (floor / ceil / trunc / round-ties-away / round-ties-even)
        assert_eq!(
            Inst::FpDataProc1 { op: FpOp1::Frintm, size: FpSize::S64, rd: V0, rn: V0 }.encode(),
            0x1E654000
        );
        assert_eq!(
            Inst::FpDataProc1 { op: FpOp1::Frintp, size: FpSize::S64, rd: V0, rn: V0 }.encode(),
            0x1E64C000
        );
        assert_eq!(
            Inst::FpDataProc1 { op: FpOp1::Frintz, size: FpSize::S64, rd: V0, rn: V0 }.encode(),
            0x1E65C000
        );
        assert_eq!(
            Inst::FpDataProc1 { op: FpOp1::Frinta, size: FpSize::S64, rd: V0, rn: V0 }.encode(),
            0x1E664000
        );
        assert_eq!(
            Inst::FpDataProc1 { op: FpOp1::Frintn, size: FpSize::S64, rd: V0, rn: V0 }.encode(),
            0x1E644000
        );
        // frintm s0, s0 (32-bit form)
        assert_eq!(
            Inst::FpDataProc1 { op: FpOp1::Frintm, size: FpSize::S32, rd: V0, rn: V0 }.encode(),
            0x1E254000
        );
        // fmadd d0, d1, d2, d3 / fmadd s0, s1, s2, s3
        assert_eq!(
            Inst::FpFma { size: FpSize::S64, rd: V0, rn: V1, rm: V2, ra: V3 }.encode(),
            0x1F420C20
        );
        assert_eq!(
            Inst::FpFma { size: FpSize::S32, rd: V0, rn: V1, rm: V2, ra: V3 }.encode(),
            0x1F020C20
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
    fn crypto_encodings() {
        // 128-bit (q) load/store; the byte offset is scaled by 16.
        assert_eq!(Inst::LoadStoreQ { load: true, rt: V0, rn: SP, offset: 0 }.encode(), 0x3DC003E0);
        assert_eq!(Inst::LoadStoreQ { load: true, rt: V1, rn: X9, offset: 16 }.encode(), 0x3DC00521);
        assert_eq!(Inst::LoadStoreQ { load: false, rt: V0, rn: SP, offset: 0 }.encode(), 0x3D8003E0);
        assert_eq!(Inst::LoadStoreQ { load: false, rt: V2, rn: X9, offset: 32 }.encode(), 0x3D800922);
        // AES (`aese`/`aesd`/`aesmc`/`aesimc v0.16b, v1.16b`).
        assert_eq!(Inst::CryptoTwo { op: CryptoTwoOp::Aese, rd: V0, rn: V1 }.encode(), 0x4E284820);
        assert_eq!(Inst::CryptoTwo { op: CryptoTwoOp::Aesd, rd: V0, rn: V1 }.encode(), 0x4E285820);
        assert_eq!(Inst::CryptoTwo { op: CryptoTwoOp::Aesmc, rd: V0, rn: V1 }.encode(), 0x4E286820);
        assert_eq!(Inst::CryptoTwo { op: CryptoTwoOp::Aesimc, rd: V0, rn: V1 }.encode(), 0x4E287820);
        // SHA-256.
        assert_eq!(Inst::CryptoThree { op: CryptoThreeOp::Sha256h, rd: V0, rn: V1, rm: V2 }.encode(), 0x5E024020);
        assert_eq!(Inst::CryptoThree { op: CryptoThreeOp::Sha256h2, rd: V0, rn: V1, rm: V2 }.encode(), 0x5E025020);
        assert_eq!(Inst::CryptoTwo { op: CryptoTwoOp::Sha256su0, rd: V0, rn: V1 }.encode(), 0x5E282820);
        assert_eq!(Inst::CryptoThree { op: CryptoThreeOp::Sha256su1, rd: V0, rn: V1, rm: V2 }.encode(), 0x5E026020);
        // SHA-1 (`sha1c`/`sha1p`/`sha1m q0, s1, v2.4s`; `sha1h s0, s1`).
        assert_eq!(Inst::CryptoThree { op: CryptoThreeOp::Sha1c, rd: V0, rn: V1, rm: V2 }.encode(), 0x5E020020);
        assert_eq!(Inst::CryptoThree { op: CryptoThreeOp::Sha1p, rd: V0, rn: V1, rm: V2 }.encode(), 0x5E021020);
        assert_eq!(Inst::CryptoThree { op: CryptoThreeOp::Sha1m, rd: V0, rn: V1, rm: V2 }.encode(), 0x5E022020);
        assert_eq!(Inst::CryptoTwo { op: CryptoTwoOp::Sha1h, rd: V0, rn: V1 }.encode(), 0x5E280820);
        assert_eq!(Inst::CryptoThree { op: CryptoThreeOp::Sha1su0, rd: V0, rn: V1, rm: V2 }.encode(), 0x5E023020);
        assert_eq!(Inst::CryptoTwo { op: CryptoTwoOp::Sha1su1, rd: V0, rn: V1 }.encode(), 0x5E281820);
        // SHA-512 (`sha512h`/`sha512h2 q0, q1, v2.2d`; `sha512su0 v0.2d, v1.2d`; `sha512su1 ...`).
        assert_eq!(Inst::CryptoThree { op: CryptoThreeOp::Sha512h, rd: V0, rn: V1, rm: V2 }.encode(), 0xCE628020);
        assert_eq!(Inst::CryptoThree { op: CryptoThreeOp::Sha512h2, rd: V0, rn: V1, rm: V2 }.encode(), 0xCE628420);
        assert_eq!(Inst::CryptoTwo { op: CryptoTwoOp::Sha512su0, rd: V0, rn: V1 }.encode(), 0xCEC08020);
        assert_eq!(Inst::CryptoThree { op: CryptoThreeOp::Sha512su1, rd: V0, rn: V1, rm: V2 }.encode(), 0xCE628820);
    }

    #[test]
    fn simd_three_encodings() {
        // Integer add across arrangements (8b/8h/4s/2d and the 64-bit `8b`).
        assert_eq!(Inst::SimdThree { op: SimdOp::Add, q: true, size: 0, rd: V0, rn: V1, rm: V2 }.encode(), 0x4E228420);
        assert_eq!(Inst::SimdThree { op: SimdOp::Add, q: true, size: 1, rd: V0, rn: V1, rm: V2 }.encode(), 0x4E628420);
        assert_eq!(Inst::SimdThree { op: SimdOp::Add, q: true, size: 2, rd: V0, rn: V1, rm: V2 }.encode(), 0x4EA28420);
        assert_eq!(Inst::SimdThree { op: SimdOp::Add, q: true, size: 3, rd: V0, rn: V1, rm: V2 }.encode(), 0x4EE28420);
        assert_eq!(Inst::SimdThree { op: SimdOp::Add, q: false, size: 0, rd: V0, rn: V1, rm: V2 }.encode(), 0x0E228420);
        // sub / mul.
        assert_eq!(Inst::SimdThree { op: SimdOp::Sub, q: true, size: 2, rd: V0, rn: V1, rm: V2 }.encode(), 0x6EA28420);
        assert_eq!(Inst::SimdThree { op: SimdOp::Mul, q: true, size: 2, rd: V0, rn: V1, rm: V2 }.encode(), 0x4EA29C20);
        // Bitwise (size field 0; `q` selects 8b/16b).
        assert_eq!(Inst::SimdThree { op: SimdOp::And, q: true, size: 0, rd: V0, rn: V1, rm: V2 }.encode(), 0x4E221C20);
        assert_eq!(Inst::SimdThree { op: SimdOp::Orr, q: true, size: 0, rd: V0, rn: V1, rm: V2 }.encode(), 0x4EA21C20);
        assert_eq!(Inst::SimdThree { op: SimdOp::Eor, q: true, size: 0, rd: V0, rn: V1, rm: V2 }.encode(), 0x6E221C20);
        // Float (size = 0 for f32, 1 for f64).
        assert_eq!(Inst::SimdThree { op: SimdOp::Fadd, q: true, size: 0, rd: V0, rn: V1, rm: V2 }.encode(), 0x4E22D420);
        assert_eq!(Inst::SimdThree { op: SimdOp::Fadd, q: true, size: 1, rd: V0, rn: V1, rm: V2 }.encode(), 0x4E62D420);
        assert_eq!(Inst::SimdThree { op: SimdOp::Fsub, q: true, size: 0, rd: V0, rn: V1, rm: V2 }.encode(), 0x4EA2D420);
        assert_eq!(Inst::SimdThree { op: SimdOp::Fmul, q: true, size: 0, rd: V0, rn: V1, rm: V2 }.encode(), 0x6E22DC20);
        assert_eq!(Inst::SimdThree { op: SimdOp::Fdiv, q: true, size: 0, rd: V0, rn: V1, rm: V2 }.encode(), 0x6E22FC20);
        assert_eq!(Inst::SimdThree { op: SimdOp::Fdiv, q: true, size: 1, rd: V0, rn: V1, rm: V2 }.encode(), 0x6E62FC20);
    }

    #[test]
    fn half_encodings() {
        // Half-precision (f16) scalar FP ops use ftype = 0b11.
        assert_eq!(Inst::FpDataProc2 { op: FpOp2::Fadd, size: FpSize::S16, rd: V0, rn: V1, rm: V2 }.encode(), 0x1EE22820);
        assert_eq!(Inst::FpDataProc2 { op: FpOp2::Fsub, size: FpSize::S16, rd: V0, rn: V1, rm: V2 }.encode(), 0x1EE23820);
        assert_eq!(Inst::FpDataProc2 { op: FpOp2::Fmul, size: FpSize::S16, rd: V0, rn: V1, rm: V2 }.encode(), 0x1EE20820);
        assert_eq!(Inst::FpDataProc2 { op: FpOp2::Fdiv, size: FpSize::S16, rd: V0, rn: V1, rm: V2 }.encode(), 0x1EE21820);
        assert_eq!(Inst::FpCmp { size: FpSize::S16, rn: V0, rm: V1 }.encode(), 0x1EE12000);
        assert_eq!(Inst::FmovFromGpr { size: FpSize::S16, rd: V0, rn: X9 }.encode(), 0x1EE70120);
        assert_eq!(Inst::LoadStoreFpUImm { load: true, size: FpSize::S16, rt: V0, rn: SP, offset: 2 }.encode(), 0x7D4007E0);
        assert_eq!(Inst::LoadStoreFpUImm { load: false, size: FpSize::S16, rt: V0, rn: SP, offset: 4 }.encode(), 0x7D000BE0);
        // fcvt to/from half.
        assert_eq!(Inst::FpCvt { from: FpSize::S16, to: FpSize::S32, rd: V0, rn: V1 }.encode(), 0x1EE24020);
        assert_eq!(Inst::FpCvt { from: FpSize::S32, to: FpSize::S16, rd: V0, rn: V1 }.encode(), 0x1E23C020);
        assert_eq!(Inst::FpCvt { from: FpSize::S16, to: FpSize::S64, rd: V0, rn: V1 }.encode(), 0x1EE2C020);
        assert_eq!(Inst::FpCvt { from: FpSize::S64, to: FpSize::S16, rd: V0, rn: V1 }.encode(), 0x1E63C020);
        // Half unary.
        assert_eq!(Inst::FpDataProc1 { op: FpOp1::Fabs, size: FpSize::S16, rd: V0, rn: V1 }.encode(), 0x1EE0C020);
        assert_eq!(Inst::FpDataProc1 { op: FpOp1::Fneg, size: FpSize::S16, rd: V0, rn: V1 }.encode(), 0x1EE14020);
        assert_eq!(Inst::FpDataProc1 { op: FpOp1::Fsqrt, size: FpSize::S16, rd: V0, rn: V1 }.encode(), 0x1EE1C020);
        assert_eq!(Inst::FpDataProc1 { op: FpOp1::Frintm, size: FpSize::S16, rd: V0, rn: V1 }.encode(), 0x1EE54020);
    }

    #[test]
    fn simd_misc_encodings() {
        // Lane-wise compares (`.4s`).
        assert_eq!(Inst::SimdThree { op: SimdOp::Cmeq, q: true, size: 2, rd: V0, rn: V1, rm: V2 }.encode(), 0x6EA28C20);
        assert_eq!(Inst::SimdThree { op: SimdOp::Cmgt, q: true, size: 2, rd: V0, rn: V1, rm: V2 }.encode(), 0x4EA23420);
        assert_eq!(Inst::SimdThree { op: SimdOp::Cmge, q: true, size: 2, rd: V0, rn: V1, rm: V2 }.encode(), 0x4EA23C20);
        assert_eq!(Inst::SimdThree { op: SimdOp::Cmhi, q: true, size: 2, rd: V0, rn: V1, rm: V2 }.encode(), 0x6EA23420);
        assert_eq!(Inst::SimdThree { op: SimdOp::Cmhs, q: true, size: 2, rd: V0, rn: V1, rm: V2 }.encode(), 0x6EA23C20);
        assert_eq!(Inst::SimdThree { op: SimdOp::Fcmeq, q: true, size: 0, rd: V0, rn: V1, rm: V2 }.encode(), 0x4E22E420);
        assert_eq!(Inst::SimdThree { op: SimdOp::Fcmgt, q: true, size: 0, rd: V0, rn: V1, rm: V2 }.encode(), 0x6EA2E420);
        assert_eq!(Inst::SimdThree { op: SimdOp::Fcmge, q: true, size: 0, rd: V0, rn: V1, rm: V2 }.encode(), 0x6E22E420);
        // Two-register misc.
        assert_eq!(Inst::SimdTwo { op: SimdUnOp::Neg, q: true, size: 2, rd: V0, rn: V1 }.encode(), 0x6EA0B820);
        assert_eq!(Inst::SimdTwo { op: SimdUnOp::Not, q: true, size: 0, rd: V0, rn: V1 }.encode(), 0x6E205820);
        assert_eq!(Inst::SimdTwo { op: SimdUnOp::Fneg, q: true, size: 0, rd: V0, rn: V1 }.encode(), 0x6EA0F820);
        assert_eq!(Inst::SimdTwo { op: SimdUnOp::Fabs, q: true, size: 0, rd: V0, rn: V1 }.encode(), 0x4EA0F820);
        assert_eq!(Inst::SimdTwo { op: SimdUnOp::Fsqrt, q: true, size: 0, rd: V0, rn: V1 }.encode(), 0x6EA1F820);
        assert_eq!(Inst::SimdTwo { op: SimdUnOp::Frintn, q: true, size: 0, rd: V0, rn: V1 }.encode(), 0x4E218820);
        assert_eq!(Inst::SimdTwo { op: SimdUnOp::Frintm, q: true, size: 0, rd: V0, rn: V1 }.encode(), 0x4E219820);
    }

    #[test]
    fn ext_reg_encodings() {
        // add x16, sp, x16 (UXTX #0) — form a frame address from a register-held offset.
        assert_eq!(
            Inst::AddSubExtReg {
                op: AddSub::Add,
                size: OperandSize::S64,
                rd: X16,
                rn: SP,
                rm: X16,
            }
            .encode(),
            0x8B3063F0
        );
        // sub sp, sp, x16 (UXTX #0) — reserve a large frame.
        assert_eq!(
            Inst::AddSubExtReg {
                op: AddSub::Sub,
                size: OperandSize::S64,
                rd: SP,
                rn: SP,
                rm: X16,
            }
            .encode(),
            0xCB3063FF
        );
        // add sp, sp, x16 (UXTX #0) — release a large frame.
        assert_eq!(
            Inst::AddSubExtReg {
                op: AddSub::Add,
                size: OperandSize::S64,
                rd: SP,
                rn: SP,
                rm: X16,
            }
            .encode(),
            0x8B3063FF
        );
        // A scaled immediate load/store offset above 32 KiB (4096 * 8) would overflow the 12-bit
        // field; the builder must route such offsets through a register instead. Confirm a
        // just-in-range offset still encodes (ldr x0, [sp, #32760]).
        assert_eq!(
            Inst::LoadStoreUImm {
                load: true,
                signed: false,
                size: MemSize::X,
                rt: X0,
                rn: SP,
                offset: 32760,
            }
            .encode(),
            0xF97FFFE0
        );
    }

    #[test]
    fn carry_encodings() {
        // adc x0, x1, x2 — add the low-word carry into the high word.
        assert_eq!(
            Inst::AddSubCarry {
                op: AddSub::Add,
                size: OperandSize::S64,
                set_flags: false,
                rd: X0,
                rn: X1,
                rm: X2,
            }
            .encode(),
            0x9A020020
        );
        // adcs x0, x1, x2 — same, setting flags (for overflow detection).
        assert_eq!(
            Inst::AddSubCarry {
                op: AddSub::Add,
                size: OperandSize::S64,
                set_flags: true,
                rd: X0,
                rn: X1,
                rm: X2,
            }
            .encode(),
            0xBA020020
        );
        // sbc x0, x1, x2 — subtract with borrow.
        assert_eq!(
            Inst::AddSubCarry {
                op: AddSub::Sub,
                size: OperandSize::S64,
                set_flags: false,
                rd: X0,
                rn: X1,
                rm: X2,
            }
            .encode(),
            0xDA020020
        );
        // sbcs x0, x1, x2 — subtract with borrow, setting flags.
        assert_eq!(
            Inst::AddSubCarry {
                op: AddSub::Sub,
                size: OperandSize::S64,
                set_flags: true,
                rd: X0,
                rn: X1,
                rm: X2,
            }
            .encode(),
            0xFA020020
        );
        // adc w5, w6, w7 — 32-bit form.
        assert_eq!(
            Inst::AddSubCarry {
                op: AddSub::Add,
                size: OperandSize::S32,
                set_flags: false,
                rd: X5,
                rn: X6,
                rm: X7,
            }
            .encode(),
            0x1A0700C5
        );
        // sbcs xzr, x1, x3 — the discard-result form used by 128-bit comparison.
        assert_eq!(
            Inst::AddSubCarry {
                op: AddSub::Sub,
                size: OperandSize::S64,
                set_flags: true,
                rd: ZR,
                rn: X1,
                rm: X3,
            }
            .encode(),
            0xFA03003F
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
        // isb sy
        assert_eq!(Inst::Isb.encode(), 0xD5033FDF);
    }
}
