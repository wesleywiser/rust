//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Float remainder has no hardware op -> `fmod`/`fmodf` libcall. Float-SIMD math (fabs) and an
//! i128 `match` (two-word compare loop) were all ICEs before. coretests exercised these.
#![crate_type = "lib"]
#![feature(portable_simd)]
use std::simd::f32x4;

// CHECK-LABEL: _rem:
// CHECK: bl _fmod
#[no_mangle]
pub fn rem(a: f64, b: f64) -> f64 {
    a % b
}

// CHECK-LABEL: _fabs4:
// CHECK: fabs
#[no_mangle]
pub fn fabs4(a: f32x4) -> f32x4 {
    use std::simd::num::SimdFloat;
    a.abs()
}

// i128 match compares both 64-bit words and branches on a combined non-zero.
// CHECK-LABEL: _pick:
// CHECK: cbnz
#[no_mangle]
pub fn pick(v: i128) -> u32 {
    match v {
        0 => 1,
        12345678901234567890 => 2,
        _ => 3,
    }
}
