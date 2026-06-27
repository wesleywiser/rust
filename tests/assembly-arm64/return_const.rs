//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Returning integer constants.
#![crate_type = "lib"]

// CHECK-LABEL: _five:
// CHECK: movz x0, #5
// CHECK: ret
#[no_mangle]
pub extern "C" fn five() -> u64 {
    5
}

// CHECK-LABEL: _zero:
// CHECK: movz x0, #0
// CHECK: ret
#[no_mangle]
pub extern "C" fn zero() -> u64 {
    0
}
