//! AArch64 register, operand-size, and condition-code definitions.
//!
//! This module is intentionally free of any `rustc_*` dependencies so that the machine-code layer
//! can be reasoned about (and unit-tested) in isolation.

/// A general-purpose register, encoded as a 5-bit field (`0..=31`).
///
/// The value `31` denotes either the zero register (`xzr`/`wzr`) or the stack pointer (`sp`)
/// depending on the instruction; callers pick the correct named constant for clarity.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Gpr(u8);

impl Gpr {
    /// Construct from a raw 5-bit encoding (`0..=31`).
    #[inline]
    pub const fn from_encoding(n: u8) -> Gpr {
        debug_assert!(n <= 31);
        Gpr(n)
    }

    /// The 5-bit register number used in instruction encodings.
    #[inline]
    pub const fn encoding(self) -> u32 {
        self.0 as u32
    }

    /// Textual name for a given operand size (`x`/`w` prefix), used by the `.s` emitter.
    ///
    /// Register `31` is rendered as the stack pointer (`sp`/`wsp`); use [`Gpr::name_zr`] in contexts
    /// where `31` denotes the zero register instead.
    pub fn name(self, size: OperandSize) -> String {
        match (self.0, size) {
            (31, OperandSize::S64) => "sp".to_string(),
            (31, OperandSize::S32) => "wsp".to_string(),
            (n, OperandSize::S64) => format!("x{n}"),
            (n, OperandSize::S32) => format!("w{n}"),
        }
    }

    /// Textual name where register `31` denotes the zero register (`xzr`/`wzr`).
    pub fn name_zr(self, size: OperandSize) -> String {
        match (self.0, size) {
            (31, _) => Gpr::zr_name(size).to_string(),
            (n, OperandSize::S64) => format!("x{n}"),
            (n, OperandSize::S32) => format!("w{n}"),
        }
    }

    /// Textual name for the zero register at a given size.
    pub fn zr_name(size: OperandSize) -> &'static str {
        match size {
            OperandSize::S64 => "xzr",
            OperandSize::S32 => "wzr",
        }
    }
}

// Named general-purpose registers.
pub const X0: Gpr = Gpr(0);
pub const X1: Gpr = Gpr(1);
pub const X2: Gpr = Gpr(2);
pub const X3: Gpr = Gpr(3);
pub const X4: Gpr = Gpr(4);
pub const X5: Gpr = Gpr(5);
pub const X6: Gpr = Gpr(6);
pub const X7: Gpr = Gpr(7);
pub const X8: Gpr = Gpr(8);
// Scratch/temporaries used by the baseline lowering (caller-saved).
pub const X9: Gpr = Gpr(9);
pub const X10: Gpr = Gpr(10);
pub const X11: Gpr = Gpr(11);
pub const X12: Gpr = Gpr(12);
pub const X13: Gpr = Gpr(13);
pub const X14: Gpr = Gpr(14);
pub const X15: Gpr = Gpr(15);
// IP0/IP1: intra-procedure-call scratch, used for large-offset materialization.
pub const X16: Gpr = Gpr(16);
pub const X17: Gpr = Gpr(17);
/// Frame pointer.
pub const FP: Gpr = Gpr(29);
/// Link register.
pub const LR: Gpr = Gpr(30);
/// Stack pointer (encodes as 31; only valid where SP is permitted).
pub const SP: Gpr = Gpr(31);
/// Zero register (encodes as 31; only valid where ZR is permitted).
pub const ZR: Gpr = Gpr(31);

/// A SIMD/floating-point register `v0..=v31`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Vreg(u8);

impl Vreg {
    #[inline]
    pub const fn from_encoding(n: u8) -> Vreg {
        debug_assert!(n <= 31);
        Vreg(n)
    }

    #[inline]
    pub const fn encoding(self) -> u32 {
        self.0 as u32
    }

    /// Textual name for a given floating-point size (`d0`, `s0`, ...).
    pub fn name(self, size: FpSize) -> String {
        format!("{}{}", size.prefix(), self.0)
    }
}

pub const V0: Vreg = Vreg(0);
pub const V1: Vreg = Vreg(1);
pub const V2: Vreg = Vreg(2);
pub const V3: Vreg = Vreg(3);
pub const V4: Vreg = Vreg(4);
pub const V5: Vreg = Vreg(5);
pub const V6: Vreg = Vreg(6);
pub const V7: Vreg = Vreg(7);
// Scratch FP registers for the baseline lowering (caller-saved).
pub const V16: Vreg = Vreg(16);
pub const V17: Vreg = Vreg(17);
pub const V18: Vreg = Vreg(18);

/// Integer operand size: 32-bit (`w`) or 64-bit (`x`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum OperandSize {
    S32,
    S64,
}

impl OperandSize {
    /// The `sf` bit (bit 31) used by most data-processing instructions.
    #[inline]
    pub const fn sf(self) -> u32 {
        match self {
            OperandSize::S32 => 0,
            OperandSize::S64 => 1,
        }
    }

    /// Size in bytes.
    #[inline]
    pub const fn bytes(self) -> u32 {
        match self {
            OperandSize::S32 => 4,
            OperandSize::S64 => 8,
        }
    }

    /// Pick a size from a width in bits, rounding `<= 32` to 32-bit and otherwise 64-bit.
    #[inline]
    pub const fn from_bits(bits: u64) -> OperandSize {
        if bits <= 32 { OperandSize::S32 } else { OperandSize::S64 }
    }
}

/// Floating-point operand size: single (`s`) or double (`d`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FpSize {
    S32,
    S64,
}

impl FpSize {
    /// The 2-bit `ftype` field used by scalar FP instructions (`00` = single, `01` = double).
    #[inline]
    pub const fn ftype(self) -> u32 {
        match self {
            FpSize::S32 => 0b00,
            FpSize::S64 => 0b01,
        }
    }

    #[inline]
    pub const fn prefix(self) -> &'static str {
        match self {
            FpSize::S32 => "s",
            FpSize::S64 => "d",
        }
    }
}

/// Condition codes (`cond` field), in their 4-bit encoding order.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Cond {
    Eq = 0b0000,
    Ne = 0b0001,
    Hs = 0b0010,
    Lo = 0b0011,
    Mi = 0b0100,
    Pl = 0b0101,
    Vs = 0b0110,
    Vc = 0b0111,
    Hi = 0b1000,
    Ls = 0b1001,
    Ge = 0b1010,
    Lt = 0b1011,
    Gt = 0b1100,
    Le = 0b1101,
    Al = 0b1110,
    Nv = 0b1111,
}

impl Cond {
    #[inline]
    pub const fn encoding(self) -> u32 {
        self as u32
    }

    /// The condition that is true exactly when `self` is false.
    #[inline]
    pub const fn invert(self) -> Cond {
        // Inverting the low bit of the 4-bit condition flips the sense of the test.
        match self {
            Cond::Eq => Cond::Ne,
            Cond::Ne => Cond::Eq,
            Cond::Hs => Cond::Lo,
            Cond::Lo => Cond::Hs,
            Cond::Mi => Cond::Pl,
            Cond::Pl => Cond::Mi,
            Cond::Vs => Cond::Vc,
            Cond::Vc => Cond::Vs,
            Cond::Hi => Cond::Ls,
            Cond::Ls => Cond::Hi,
            Cond::Ge => Cond::Lt,
            Cond::Lt => Cond::Ge,
            Cond::Gt => Cond::Le,
            Cond::Le => Cond::Gt,
            Cond::Al => Cond::Nv,
            Cond::Nv => Cond::Al,
        }
    }

    pub const fn mnemonic(self) -> &'static str {
        match self {
            Cond::Eq => "eq",
            Cond::Ne => "ne",
            Cond::Hs => "hs",
            Cond::Lo => "lo",
            Cond::Mi => "mi",
            Cond::Pl => "pl",
            Cond::Vs => "vs",
            Cond::Vc => "vc",
            Cond::Hi => "hi",
            Cond::Ls => "ls",
            Cond::Ge => "ge",
            Cond::Lt => "lt",
            Cond::Gt => "gt",
            Cond::Le => "le",
            Cond::Al => "al",
            Cond::Nv => "nv",
        }
    }
}
