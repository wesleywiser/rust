//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Sub-word integers (`i8`/`i16`) live zero-extended in registers (loaded with `ldrb`/`ldrh`), so
//! width-sensitive *signed* operations must sign-extend first or they compute on the wrong value
//! (e.g. `-5i8` would compare as `251`). Signed compares, divides, and arithmetic shifts therefore
//! emit `sxtb`/`sxth`; unsigned operations must NOT (zero extension is already correct).

#![crate_type = "lib"]

// Signed `<`: both operands are sign-extended before the compare.
// CHECK-LABEL: _lt_i8:
// CHECK: ldrb w9, [sp, #0]
// CHECK-NEXT: sxtb w9, w9
// CHECK-NEXT: ldrb w10, [sp, #1]
// CHECK-NEXT: sxtb w10, w10
// CHECK-NEXT: cmp w9, w10
// CHECK-NEXT: csinc w9, wzr, wzr, ge
#[no_mangle]
pub extern "C" fn lt_i8(a: i8, b: i8) -> bool {
    a < b
}

// Unsigned `<`: the loads feed the compare directly, with no `sxtb` in between.
// CHECK-LABEL: _ult_u8:
// CHECK: ldrb w9, [sp, #0]
// CHECK-NEXT: ldrb w10, [sp, #1]
// CHECK-NEXT: cmp w9, w10
// CHECK-NEXT: csinc w9, wzr, wzr, hs
#[no_mangle]
pub extern "C" fn ult_u8(a: u8, b: u8) -> bool {
    a < b
}

// Arithmetic shift right: the shifted value is sign-extended; the shift amount is loaded plainly.
// CHECK-LABEL: _asr_i16:
// CHECK: sxth w9, w9
// CHECK-NEXT: ldrh w10,
// CHECK-NEXT: asr w9, w9, w10
#[no_mangle]
pub extern "C" fn asr_i16(a: i16, b: i16) -> i16 {
    a >> b
}
