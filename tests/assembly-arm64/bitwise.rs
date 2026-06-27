//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Bitwise operations.
#![crate_type = "lib"]

// CHECK-LABEL: _bit_and:
// CHECK: and x9, x9, x10
#[no_mangle]
pub extern "C" fn bit_and(a: u64, b: u64) -> u64 {
    a & b
}

// CHECK-LABEL: _bit_or:
// CHECK: orr x9, x9, x10
#[no_mangle]
pub extern "C" fn bit_or(a: u64, b: u64) -> u64 {
    a | b
}

// CHECK-LABEL: _bit_xor:
// CHECK: eor x9, x9, x10
#[no_mangle]
pub extern "C" fn bit_xor(a: u64, b: u64) -> u64 {
    a ^ b
}
