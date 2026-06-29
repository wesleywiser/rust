//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Lane-wise SIMD comparisons, negation, and floating-point unary ops lower to native NEON
//! instructions for 64/128-bit vectors: `cmeq`/`cmgt`/`cmge`/`cmhi`/`cmhs` (and `fcmeq`/`fcmgt`/
//! `fcmge`) for compares, `neg`/`fneg`/`fabs`/`fsqrt`/`frint*` for the unary ops. The `ne` forms are
//! a compare followed by a bitwise `not`, and the `<`/`<=` forms reuse the `>`/`>=` instruction with
//! swapped operands.
#![crate_type = "lib"]
#![feature(core_intrinsics)]
#![feature(portable_simd)]
use std::intrinsics::simd::{
    simd_ceil, simd_eq, simd_fabs, simd_floor, simd_fsqrt, simd_ge, simd_gt, simd_le, simd_lt,
    simd_ne, simd_neg, simd_round, simd_trunc,
};
use std::simd::{f32x4, f64x2, i32x4, u32x4};

// Integer equality is a single `cmeq`.
// CHECK-LABEL: _eq_i32:
// CHECK: cmeq v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
#[no_mangle]
pub unsafe extern "C" fn eq_i32(a: i32x4, b: i32x4) -> i32x4 {
    simd_eq(a, b)
}

// Inequality is `cmeq` then a bitwise `not`.
// CHECK-LABEL: _ne_i32:
// CHECK: cmeq v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
// CHECK: not v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
#[no_mangle]
pub unsafe extern "C" fn ne_i32(a: i32x4, b: i32x4) -> i32x4 {
    simd_ne(a, b)
}

// Signed greater-than is `cmgt`; signed greater-or-equal is `cmge`.
// CHECK-LABEL: _gt_i32:
// CHECK: cmgt v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
#[no_mangle]
pub unsafe extern "C" fn gt_i32(a: i32x4, b: i32x4) -> i32x4 {
    simd_gt(a, b)
}

// CHECK-LABEL: _ge_i32:
// CHECK: cmge v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
#[no_mangle]
pub unsafe extern "C" fn ge_i32(a: i32x4, b: i32x4) -> i32x4 {
    simd_ge(a, b)
}

// Signed less-than reuses `cmgt` with swapped operands (no separate instruction).
// CHECK-LABEL: _lt_i32:
// CHECK: cmgt v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
#[no_mangle]
pub unsafe extern "C" fn lt_i32(a: i32x4, b: i32x4) -> i32x4 {
    simd_lt(a, b)
}

// Unsigned greater-than/greater-or-equal use `cmhi`/`cmhs`.
// CHECK-LABEL: _gt_u32:
// CHECK: cmhi v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
#[no_mangle]
pub unsafe extern "C" fn gt_u32(a: u32x4, b: u32x4) -> u32x4 {
    simd_gt(a, b)
}

// CHECK-LABEL: _le_u32:
// CHECK: cmhs v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
#[no_mangle]
pub unsafe extern "C" fn le_u32(a: u32x4, b: u32x4) -> u32x4 {
    simd_le(a, b)
}

// Float compares use the `fcm*` family (ordered).
// CHECK-LABEL: _eq_f32:
// CHECK: fcmeq v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
#[no_mangle]
pub unsafe extern "C" fn eq_f32(a: f32x4, b: f32x4) -> i32x4 {
    simd_eq(a, b)
}

// CHECK-LABEL: _gt_f32:
// CHECK: fcmgt v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
#[no_mangle]
pub unsafe extern "C" fn gt_f32(a: f32x4, b: f32x4) -> i32x4 {
    simd_gt(a, b)
}

// CHECK-LABEL: _ge_f32:
// CHECK: fcmge v{{[0-9]+}}.4s, v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
#[no_mangle]
pub unsafe extern "C" fn ge_f32(a: f32x4, b: f32x4) -> i32x4 {
    simd_ge(a, b)
}

// Integer negation is a single `neg`.
// CHECK-LABEL: _neg_i32:
// CHECK: neg v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
#[no_mangle]
pub unsafe extern "C" fn neg_i32(a: i32x4) -> i32x4 {
    simd_neg(a)
}

// Float negation/abs/sqrt are `fneg`/`fabs`/`fsqrt`.
// CHECK-LABEL: _neg_f32:
// CHECK: fneg v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
#[no_mangle]
pub unsafe extern "C" fn neg_f32(a: f32x4) -> f32x4 {
    simd_neg(a)
}

// CHECK-LABEL: _abs_f64:
// CHECK: fabs v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
#[no_mangle]
pub unsafe extern "C" fn abs_f64(a: f64x2) -> f64x2 {
    simd_fabs(a)
}

// CHECK-LABEL: _sqrt_f32:
// CHECK: fsqrt v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
#[no_mangle]
pub unsafe extern "C" fn sqrt_f32(a: f32x4) -> f32x4 {
    simd_fsqrt(a)
}

// Rounding maps to the `frint*` variants: ceil -> frintp, floor -> frintm, round -> frinta,
// trunc -> frintz.
// CHECK-LABEL: _ceil_f32:
// CHECK: frintp v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
#[no_mangle]
pub unsafe extern "C" fn ceil_f32(a: f32x4) -> f32x4 {
    simd_ceil(a)
}

// CHECK-LABEL: _floor_f32:
// CHECK: frintm v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
#[no_mangle]
pub unsafe extern "C" fn floor_f32(a: f32x4) -> f32x4 {
    simd_floor(a)
}

// CHECK-LABEL: _round_f32:
// CHECK: frinta v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
#[no_mangle]
pub unsafe extern "C" fn round_f32(a: f32x4) -> f32x4 {
    simd_round(a)
}

// CHECK-LABEL: _trunc_f32:
// CHECK: frintz v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
#[no_mangle]
pub unsafe extern "C" fn trunc_f32(a: f32x4) -> f32x4 {
    simd_trunc(a)
}
