//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Materializing an immediate into a 32-bit (`W`) register must only use `movk` shifts of 0 and 16:
//! the `hw` field is a single bit for 32-bit move-wide, so `movk w, #imm, lsl #32` (or `#48`) is an
//! illegal encoding that faults as SIGILL at runtime. A negative 32-bit constant is the trap,
//! because its wide (`u128`/`u64`) representation is sign-extended to all-ones; the backend must
//! mask to the destination width and stop after the `lsl #16` chunk.

#![crate_type = "lib"]

// `-1` loaded directly: exactly two move-wide instructions, then the return — the `CHECK-NEXT`
// chain proves no `lsl #32`/`lsl #48` chunk is emitted after the `lsl #16` one.
// CHECK-LABEL: _neg_one:
// CHECK: movz w0, #65535
// CHECK-NEXT: movk w0, #65535, lsl #16
// CHECK-NEXT: ldp x29, x30, [sp], #16
// CHECK-NEXT: ret
#[no_mangle]
pub extern "C" fn neg_one() -> i32 {
    core::hint::black_box(-1i32)
}

// `x ^ -1` (bitwise NOT): the `-1` operand is materialized into `w10` with the same two-instruction
// sequence and is immediately consumed by the `eor`, again pinning that no wider shift is emitted.
// CHECK-LABEL: _not_i32:
// CHECK: ldr w9, [sp, #0]
// CHECK-NEXT: movz w10, #65535
// CHECK-NEXT: movk w10, #65535, lsl #16
// CHECK-NEXT: eor w9, w9, w10
#[no_mangle]
pub extern "C" fn not_i32(x: i32) -> i32 {
    x ^ -1
}
