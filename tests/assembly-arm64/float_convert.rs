//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Conversions between integers and floating-point, and between float widths.
//!
//! - integer -> float: `scvtf`/`ucvtf` (the source is first widened to 64 bits)
//! - float -> integer: `fcvtzs`/`fcvtzu` (round toward zero, saturating)
//! - float width: `fcvt`
#![crate_type = "lib"]

// `i64 -> f64`: `scvtf` from a 64-bit GPR.
// CHECK-LABEL: _i2d:
// CHECK: scvtf d16, x9
// CHECK-NEXT: str d16, [sp, #16]
// CHECK-NEXT: ldr d0, [sp, #16]
#[no_mangle]
pub extern "C" fn i2d(x: i64) -> f64 {
    x as f64
}

// `u64 -> f64`: unsigned `ucvtf`.
// CHECK-LABEL: _u2d:
// CHECK: ucvtf d16, x9
// CHECK-NEXT: str d16, [sp, #16]
#[no_mangle]
pub extern "C" fn u2d(x: u64) -> f64 {
    x as f64
}

// `i32 -> f32`: the source is first sign-extended to 64 bits (shift left then arithmetic-shift
// right by 32) before `scvtf s, x`.
// CHECK-LABEL: _i2f:
// CHECK: ldr w9, [sp, #0]
// CHECK-NEXT: movz x10, #32
// CHECK-NEXT: lsl x9, x9, x10
// CHECK-NEXT: asr x9, x9, x10
// CHECK-NEXT: str x9, [sp, #8]
// CHECK-NEXT: ldr x9, [sp, #8]
// CHECK-NEXT: scvtf s16, x9
#[no_mangle]
pub extern "C" fn i2f(x: i32) -> f32 {
    x as f32
}

// `f64 -> i64`: round-toward-zero `fcvtzs` (saturating in hardware).
// CHECK-LABEL: _d2i:
// CHECK: ldr d16, [sp, #0]
// CHECK-NEXT: fcvtzs x9, d16
// CHECK-NEXT: str x9, [sp, #8]
#[no_mangle]
pub extern "C" fn d2i(x: f64) -> i64 {
    x as i64
}

// `f64 -> u64`: unsigned `fcvtzu`.
// CHECK-LABEL: _d2u:
// CHECK: ldr d16, [sp, #0]
// CHECK-NEXT: fcvtzu x9, d16
// CHECK-NEXT: str x9, [sp, #8]
#[no_mangle]
pub extern "C" fn d2u(x: f64) -> u64 {
    x as u64
}

// `f32 -> f64`: widening `fcvt d, s`.
// CHECK-LABEL: _f2d:
// CHECK: ldr s16, [sp, #0]
// CHECK-NEXT: fcvt d16, s16
// CHECK-NEXT: str d16, [sp, #8]
#[no_mangle]
pub extern "C" fn f2d(x: f32) -> f64 {
    x as f64
}

// `f64 -> f32`: narrowing `fcvt s, d`.
// CHECK-LABEL: _d2f:
// CHECK: ldr d16, [sp, #0]
// CHECK-NEXT: fcvt s16, d16
// CHECK-NEXT: str s16, [sp, #8]
#[no_mangle]
pub extern "C" fn d2f(x: f64) -> f32 {
    x as f32
}
