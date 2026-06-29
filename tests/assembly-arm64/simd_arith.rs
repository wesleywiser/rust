//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Lane-wise SIMD arithmetic (`simd_add`/`simd_mul`) and a wide-lane `simd_swizzle` (4-byte lanes,
//! not just bytes). These were previously unimplemented: `simd_add must be overridden` and "only
//! byte-lane SIMD shuffles are supported". rand's chacha20 backend needs both.
#![crate_type = "lib"]
#![feature(portable_simd)]
use std::simd::{simd_swizzle, u32x4};

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
