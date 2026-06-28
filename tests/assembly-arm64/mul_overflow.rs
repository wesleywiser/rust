//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! The `mul_with_overflow` intrinsic detects overflow regardless of the `-Coverflow-checks` flag.
//! Calling it directly keeps the check inline (the `overflowing_mul` wrapper would hide it in a
//! separate function). 64-bit multiplies take the high half with `umulh`/`smulh` and compare it
//! against zero (unsigned) or the sign of the low half (signed); narrower widths widen to a single
//! 64-bit multiply and range-check the result. Each exact sequence is pinned.
#![feature(core_intrinsics)]
#![crate_type = "lib"]

// Unsigned 64-bit: overflow iff the high half is non-zero.
// CHECK-LABEL: _umul:
// CHECK: umulh x12, x9, x10
// CHECK-NEXT: mul x9, x9, x10
// CHECK-NEXT: cmp x12, xzr
// CHECK-NEXT: csinc w11, wzr, wzr, eq
#[no_mangle]
pub extern "C" fn umul(a: u64, b: u64) -> bool {
    core::intrinsics::mul_with_overflow(a, b).1
}

// Signed 64-bit: overflow iff the high half differs from the sign-extension of the low half.
// CHECK-LABEL: _imul:
// CHECK: smulh x12, x9, x10
// CHECK-NEXT: mul x9, x9, x10
// CHECK-NEXT: movz x13, #63
// CHECK-NEXT: asr x13, x9, x13
// CHECK-NEXT: cmp x12, x13
// CHECK-NEXT: csinc w11, wzr, wzr, eq
#[no_mangle]
pub extern "C" fn imul(a: i64, b: i64) -> bool {
    core::intrinsics::mul_with_overflow(a, b).1
}

// Unsigned 32-bit: widen to 64 bits, multiply once, overflow iff any bit above bit 31 is set.
// CHECK-LABEL: _u32mul:
// CHECK: mul x9, x9, x10
// CHECK-NEXT: movz x13, #32
// CHECK-NEXT: lsr x12, x9, x13
// CHECK-NEXT: cmp x12, xzr
// CHECK-NEXT: csinc w11, wzr, wzr, eq
#[no_mangle]
pub extern "C" fn u32mul(a: u32, b: u32) -> bool {
    core::intrinsics::mul_with_overflow(a, b).1
}

// Signed 32-bit: widen, multiply, overflow iff the product differs from the sign-extension of its
// low 32 bits.
// CHECK-LABEL: _i32mul:
// CHECK: mul x9, x9, x10
// CHECK-NEXT: movz x13, #32
// CHECK-NEXT: lsl x12, x9, x13
// CHECK-NEXT: asr x12, x12, x13
// CHECK-NEXT: cmp x9, x12
// CHECK-NEXT: csinc w11, wzr, wzr, eq
#[no_mangle]
pub extern "C" fn i32mul(a: i32, b: i32) -> bool {
    core::intrinsics::mul_with_overflow(a, b).1
}

