//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Overflow-checked add/sub (`overflowing_add`, the `+`/`-` operators under overflow checks, ...)
//! must detect overflow at the operand's *own* width. The NZCV flags only describe 32/64-bit
//! overflow, so for `i8`/`i16` the operands are widened, the sum/difference computed (which then
//! cannot itself overflow), and the result range-checked against the type's bounds. Full-width
//! types keep using the hardware flags directly.

#![crate_type = "lib"]

// `i8` add: sign-extend, add, then range-check against the i8 bounds (127 and -128).
// CHECK-LABEL: _oadd_i8:
// CHECK: sxtb w9, w9
// CHECK: sxtb w10, w10
// CHECK: add w9, w9, w10
// CHECK: movz w10, #127
// CHECK: cmp w9, w10
#[no_mangle]
pub extern "C" fn oadd_i8(a: i8, b: i8) -> (i8, bool) {
    a.overflowing_add(b)
}

// `i32` add: the type spans the operation width, so the flag-setting `adds` is used directly.
// CHECK-LABEL: _oadd_i32:
// CHECK: adds w9, w9, w10
#[no_mangle]
pub extern "C" fn oadd_i32(a: i32, b: i32) -> (i32, bool) {
    a.overflowing_add(b)
}
