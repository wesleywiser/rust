//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! `f128` negation must flip the sign bit (bit 127), which lives in the high 64-bit word. A native
//! scalar `fneg` would treat the 128-bit value as an `f64` and corrupt it (negating only the low
//! word and dropping the high one), so negation operates on the two GPR words: the high word is
//! XORed with `0x8000_0000_0000_0000`.
#![feature(f128)]
#![crate_type = "lib"]

// CHECK-LABEL: _negate:
// The 128-bit operand arrives in q0; split it into lo (x9) / hi (x10) words.
// CHECK:      ldr x9, [sp
// CHECK:      ldr x10, [sp
// Build the sign-bit mask 0x8000000000000000 and flip bit 127 in the HIGH word.
// CHECK:      movk x11, #32768, lsl #48
// CHECK:      eor x10, x10, x11
// CHECK-NOT:  fneg
// CHECK:      ret
#[no_mangle]
pub extern "C" fn negate(x: f128) -> f128 {
    -x
}
