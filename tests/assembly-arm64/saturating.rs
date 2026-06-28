//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! `saturating_add`/`saturating_sub` are `must_be_overridden`. They lower to the wrapping operation
//! followed by an overflow test and a conditional select of the saturation bound, rather than a
//! call or a fallback. At 64 bits the unsigned forms use the carry/borrow directly (the sum is below
//! an operand on wrap; the difference borrows when `a < b`).

#![crate_type = "lib"]

// Unsigned 64-bit add: compute `a + b`, then `csel` the all-ones max on carry.
// CHECK-LABEL: _sat_add_u64:
// CHECK: add x9, x9, x10
// CHECK: movz x10, #65535
// CHECK: movk x10, #65535, lsl #48
// CHECK: csel
#[no_mangle]
pub extern "C" fn sat_add_u64(a: u64, b: u64) -> u64 {
    a.saturating_add(b)
}

// Unsigned subtract: compute `a - b`, then `csel` zero when it borrowed.
// CHECK-LABEL: _sat_sub_u32:
// CHECK: sub x9, x9, x10
// CHECK: csel
#[no_mangle]
pub extern "C" fn sat_sub_u32(a: u32, b: u32) -> u32 {
    a.saturating_sub(b)
}
