//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Floating-point math intrinsics. Where AArch64 has an instruction whose IEEE-754 semantics
//! match exactly, the intrinsic lowers to that single instruction: `sqrt` -> `fsqrt`, `fabs` ->
//! `fabs`, the four directed roundings -> `frint{m,p,z}` and the two round-to-nearest modes ->
//! `frint{a,n}`, and fused multiply-add -> `fmadd`. The transcendental routines and `copysign`
//! (which have no exact single-instruction form) call the corresponding libm routine, and integer
//! power calls the compiler-builtins helper `__powidf2`.
#![feature(core_intrinsics)]
#![crate_type = "lib"]

// CHECK-LABEL: _f_sqrt:
// CHECK: fsqrt d16, d16
#[no_mangle]
pub extern "C" fn f_sqrt(x: f64) -> f64 {
    core::intrinsics::sqrtf64(x)
}

// `sqrtf32` uses the single-precision form (`s` registers).
// CHECK-LABEL: _f_sqrtf32:
// CHECK: fsqrt s16, s16
#[no_mangle]
pub extern "C" fn f_sqrtf32(x: f32) -> f32 {
    core::intrinsics::sqrtf32(x)
}

// `f64::abs` is the generic `fabs` intrinsic; for f64 it is a single `fabs` instruction.
// CHECK-LABEL: _f_abs:
// CHECK: fabs d16, d16
#[no_mangle]
pub extern "C" fn f_abs(x: f64) -> f64 {
    x.abs()
}

// floor -> round toward -inf.
// CHECK-LABEL: _f_floor:
// CHECK: frintm d16, d16
#[no_mangle]
pub extern "C" fn f_floor(x: f64) -> f64 {
    core::intrinsics::floorf64(x)
}

// ceil -> round toward +inf.
// CHECK-LABEL: _f_ceil:
// CHECK: frintp d16, d16
#[no_mangle]
pub extern "C" fn f_ceil(x: f64) -> f64 {
    core::intrinsics::ceilf64(x)
}

// trunc -> round toward zero.
// CHECK-LABEL: _f_trunc:
// CHECK: frintz d16, d16
#[no_mangle]
pub extern "C" fn f_trunc(x: f64) -> f64 {
    core::intrinsics::truncf64(x)
}

// round -> round to nearest, ties away from zero.
// CHECK-LABEL: _f_round:
// CHECK: frinta d16, d16
#[no_mangle]
pub extern "C" fn f_round(x: f64) -> f64 {
    core::intrinsics::roundf64(x)
}

// round_ties_even -> round to nearest, ties to even.
// CHECK-LABEL: _f_rte:
// CHECK: frintn d16, d16
#[no_mangle]
pub extern "C" fn f_rte(x: f64) -> f64 {
    core::intrinsics::round_ties_even_f64(x)
}

// Fused multiply-add: a single `fmadd` (one rounding).
// CHECK-LABEL: _f_fma:
// CHECK: fmadd d16, d16, d17, d0
#[no_mangle]
pub extern "C" fn f_fma(a: f64, b: f64, c: f64) -> f64 {
    core::intrinsics::fmaf64(a, b, c)
}

// Transcendental routines call libm.
// CHECK-LABEL: _f_sin:
// CHECK: bl _sin
#[no_mangle]
pub extern "C" fn f_sin(x: f64) -> f64 {
    core::intrinsics::sinf64(x)
}

// CHECK-LABEL: _f_pow:
// CHECK: bl _pow
#[no_mangle]
pub extern "C" fn f_pow(a: f64, b: f64) -> f64 {
    core::intrinsics::powf64(a, b)
}

// `copysign` has no exact single instruction, so it also calls libm.
// CHECK-LABEL: _f_copysign:
// CHECK: bl _copysign
#[no_mangle]
pub extern "C" fn f_copysign(a: f64, b: f64) -> f64 {
    core::intrinsics::copysignf64(a, b)
}

// Integer power calls the compiler-builtins helper.
// CHECK-LABEL: _f_powi:
// CHECK: bl ___powidf2
#[no_mangle]
pub extern "C" fn f_powi(a: f64, b: i32) -> f64 {
    core::intrinsics::powif64(a, b)
}
