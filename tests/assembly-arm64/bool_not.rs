//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Logical/bitwise NOT must complement within the operand's own bit width. For `bool` (`i1`) that
//! means `!b == b ^ 1`, keeping the result in `{0, 1}`: xoring with a full-width all-ones and
//! storing into the 1-byte slot would leave `0xfe`, which reads as truthy and miscompiles every
//! `!bool` (this surfaced as `RangeInclusive::is_empty` always returning `true`, so `for _ in 1..=n`
//! looped zero times). Wider integers still complement across their whole width.

#![crate_type = "lib"]

// `!bool` xors with `#1` only — note the byte (`strb`/`ldrb`) accesses and the single `movz #1`.
// CHECK-LABEL: _not_bool:
// CHECK: ldrb w9, [sp, #0]
// CHECK-NEXT: movz w10, #1
// CHECK-NEXT: eor w9, w9, w10
// CHECK-NEXT: strb w9, [sp, #1]
#[no_mangle]
pub extern "C" fn not_bool(b: bool) -> bool {
    !b
}

// `!i32` still complements the full 32 bits: the mask is `0xffffffff` (movz/movk), not `1`.
// CHECK-LABEL: _not_i32:
// CHECK: ldr w9, [sp, #0]
// CHECK-NEXT: movz w10, #65535
// CHECK-NEXT: movk w10, #65535, lsl #16
// CHECK-NEXT: eor w9, w9, w10
#[no_mangle]
pub extern "C" fn not_i32(x: i32) -> i32 {
    !x
}
