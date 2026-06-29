//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Lane-wise SIMD arithmetic (`simd_add`/`simd_mul`) and a wide-lane `simd_swizzle` (4-byte lanes,
//! not just bytes). These were previously unimplemented: `simd_add must be overridden` and "only
//! byte-lane SIMD shuffles are supported". rand's chacha20 backend needs both.
#![crate_type = "lib"]
#![feature(core_intrinsics)]
#![feature(portable_simd)]
use std::intrinsics::simd::{simd_div, simd_rem};
use std::simd::{f32x4, i32x4, simd_swizzle, u32x4};

// CHECK-LABEL: _add4:
// CHECK: add x{{[0-9]+}}
#[no_mangle]
pub fn add4(a: u32x4, b: u32x4) -> u32x4 {
    a + b
}

// A wide-lane shuffle multiplies the index by the 4-byte lane size (ldr/str of a word per lane).
// CHECK-LABEL: _shuf4:
// CHECK: ldr w{{[0-9]+}}
// CHECK: ret
#[no_mangle]
pub fn shuf4(a: u32x4, b: u32x4) -> u32x4 {
    simd_swizzle!(a, b, [0, 5, 2, 7])
}

// Integer SIMD divide has no NEON instruction: it is a per-lane `udiv`/`sdiv` (computed in a
// 64-bit GPR after a zero/sign-extending load).
// CHECK-LABEL: _divu:
// CHECK: udiv x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
#[no_mangle]
pub unsafe extern "C" fn divu(a: u32x4, b: u32x4) -> u32x4 {
    simd_div(a, b)
}

// CHECK-LABEL: _divi:
// CHECK: sdiv x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
#[no_mangle]
pub unsafe extern "C" fn divi(a: i32x4, b: i32x4) -> i32x4 {
    simd_div(a, b)
}

// Remainder is `udiv` then `msub` (r = a - (a / b) * b) per lane.
// CHECK-LABEL: _remu:
// CHECK: udiv x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: msub x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
#[no_mangle]
pub unsafe extern "C" fn remu(a: u32x4, b: u32x4) -> u32x4 {
    simd_rem(a, b)
}

// Float SIMD divide uses the FP unit; float remainder calls `fmodf` per lane.
// CHECK-LABEL: _divf:
// CHECK: fdiv s{{[0-9]+}}, s{{[0-9]+}}, s{{[0-9]+}}
#[no_mangle]
pub unsafe extern "C" fn divf(a: f32x4, b: f32x4) -> f32x4 {
    simd_div(a, b)
}

// CHECK-LABEL: _remf:
// CHECK: bl _fmodf
#[no_mangle]
pub unsafe extern "C" fn remf(a: f32x4, b: f32x4) -> f32x4 {
    simd_rem(a, b)
}
