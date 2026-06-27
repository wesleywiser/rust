//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! `overflowing_mul` uses the `mul_with_overflow` intrinsic, which detects overflow regardless of
//! the `-Coverflow-checks` flag. 64-bit multiplies use `umulh`/`smulh`; narrower widths widen to a
//! single 64-bit multiply and range-check the result.
#![crate_type = "lib"]

// CHECK-LABEL: _umul_ovf:
// CHECK: umulh
// CHECK: mul x
#[no_mangle]
pub extern "C" fn umul_ovf(a: u64, b: u64) -> bool {
    a.overflowing_mul(b).1
}

// CHECK-LABEL: _imul_ovf:
// CHECK: smulh
// CHECK: mul x
#[no_mangle]
pub extern "C" fn imul_ovf(a: i64, b: i64) -> bool {
    a.overflowing_mul(b).1
}

// CHECK-LABEL: _u32mul_ovf:
// CHECK: mul x
#[no_mangle]
pub extern "C" fn u32mul_ovf(a: u32, b: u32) -> bool {
    a.overflowing_mul(b).1
}
