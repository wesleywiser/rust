//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! 128-bit integers are modelled as a low/high pair of 64-bit words in a 16-byte frame slot, passed
//! in two consecutive registers (x0:x1, x2:x3) and returned in x0:x1. Arithmetic that has a compact
//! form is emitted inline (carry-chained add/sub, schoolbook multiply, flag-based comparison) and
//! the rest goes to compiler-builtins libcalls.
#![crate_type = "lib"]

// Add propagates the carry from the low word into the high word.
// CHECK-LABEL: _add:
// CHECK: adds x9, x9, x11
// CHECK-NEXT: adc x10, x10, x12
#[no_mangle]
pub extern "C" fn add(a: u128, b: u128) -> u128 {
    a.wrapping_add(b)
}

// Multiply is the schoolbook 64-bit partial-product expansion (no libcall): the high word is the
// high half of the low product plus the two cross products.
// CHECK-LABEL: _mul:
// CHECK: umulh x13, x9, x11
// CHECK-NEXT: madd x13, x10, x11, x13
// CHECK-NEXT: madd x13, x9, x12, x13
// CHECK-NEXT: mul x9, x9, x11
#[no_mangle]
pub extern "C" fn mul(a: u128, b: u128) -> u128 {
    a.wrapping_mul(b)
}

// Unsigned less-than does a full 128-bit subtract and reads the carry flag (`cset lo`, encoded as
// `csinc ..., hs`).
// CHECK-LABEL: _ult:
// CHECK: cmp x9, x11
// CHECK-NEXT: sbcs xzr, x10, x12
// CHECK-NEXT: csinc w9, wzr, wzr, hs
#[no_mangle]
pub extern "C" fn ult(a: u128, b: u128) -> bool {
    a < b
}

// Division has no inline form; it calls the compiler-builtins libcall.
// CHECK-LABEL: _udiv:
// CHECK: bl ___udivti3
#[no_mangle]
pub extern "C" fn udiv(a: u128, b: u128) -> u128 {
    a / b
}
