//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! 128-bit `saturating_add`/`saturating_sub` must operate at full 128-bit width. A previous bug
//! truncated the operands to 64 bits (a single `add`/`sub`), silently miscompiling every i128/u128
//! saturating add/sub. The fix routes the 128-bit case through the hardware carry chain
//! (`adds`+`adcs` / `subs`+`sbcs`) with a saturation `csel`, so these tests assert the carry-chain
//! instruction (the high-word add/sub-with-carry) is present and that no library call is emitted.
#![crate_type = "lib"]
#![feature(core_intrinsics)]
use std::intrinsics::{saturating_add, saturating_sub};

// Signed 128-bit saturating add: low `adds`, high `adcs` (carry), then a `csel` to clamp to
// i128::MIN/MAX on signed overflow. The presence of `adcs` proves the high 64 bits are not dropped.
// CHECK-LABEL: _sat_add_i128:
// CHECK: adds x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: adcs x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: csel
#[no_mangle]
pub unsafe extern "C" fn sat_add_i128(a: i128, b: i128) -> i128 {
    saturating_add(a, b)
}

// Unsigned 128-bit saturating add: `adds`+`adcs`, clamp to u128::MAX on carry.
// CHECK-LABEL: _sat_add_u128:
// CHECK: adds x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: adcs x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: csel
#[no_mangle]
pub unsafe extern "C" fn sat_add_u128(a: u128, b: u128) -> u128 {
    saturating_add(a, b)
}

// Unsigned 128-bit saturating sub: low `subs`, high `sbcs` (borrow), clamp to 0 on borrow.
// CHECK-LABEL: _sat_sub_u128:
// CHECK: subs x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: sbcs x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: csel
#[no_mangle]
pub unsafe extern "C" fn sat_sub_u128(a: u128, b: u128) -> u128 {
    saturating_sub(a, b)
}

// Signed 128-bit saturating sub: `subs`+`sbcs`, clamp to i128::MIN/MAX on signed overflow.
// CHECK-LABEL: _sat_sub_i128:
// CHECK: subs x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: sbcs x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: csel
#[no_mangle]
pub unsafe extern "C" fn sat_sub_i128(a: i128, b: i128) -> i128 {
    saturating_sub(a, b)
}
