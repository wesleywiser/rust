//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! `ctpop` (population count) is a `must_be_overridden` intrinsic, so the backend has to lower it
//! directly (a `Fallback` ICEs, like `black_box`). AArch64 has no scalar popcount, so it is lowered
//! with the classic SWAR sequence in 64 bits: mask/shift/add reductions with the magic constants
//! `0x5555…`, `0x3333…`, `0x0f0f…`, then a multiply by `0x0101…` and a `>> 56`. This also exercises
//! the 64-bit `movk` materialization (all four `hw` shifts are legal for an `X` register).

#![crate_type = "lib"]

// CHECK-LABEL: _pc64:
// The first reduction mask `0x5555555555555555` is built with all four legal 64-bit `movk` shifts.
// CHECK: movz x10, #21845
// CHECK-NEXT: movk x10, #21845, lsl #16
// CHECK-NEXT: movk x10, #21845, lsl #32
// CHECK-NEXT: movk x10, #21845, lsl #48
// CHECK: sub x9, x9, x10
// Second reduction with `0x3333333333333333`, then the accumulate.
// CHECK: movz x10, #13107
// CHECK: add x9, x9, x10
// Nibble mask `0x0f0f0f0f0f0f0f0f`.
// CHECK: movz x10, #3855
// Multiply by `0x0101010101010101` and take the top byte (`>> 56`).
// CHECK: movz x10, #257
// CHECK: mul x9, x9, x10
// CHECK: movz x10, #56
// CHECK-NEXT: lsr x9, x9, x10
#[no_mangle]
pub extern "C" fn pc64(x: u64) -> u32 {
    x.count_ones()
}
