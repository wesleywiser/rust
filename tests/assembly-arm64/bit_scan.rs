//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! `ctlz`/`cttz` are `must_be_overridden` intrinsics. AArch64 has `clz` directly, so
//! `u64::leading_zeros` is a single `clz`. There is no trailing-zero instruction, so
//! `trailing_zeros` is `clz(rbit(x))` — bit-reverse, then count leading zeros.

#![crate_type = "lib"]

// A 64-bit `leading_zeros` is exactly one `clz` (no width adjustment is needed at 64 bits).
// CHECK-LABEL: _lz64:
// CHECK: ldr x9, [sp, #0]
// CHECK-NEXT: clz x9, x9
#[no_mangle]
pub extern "C" fn lz64(x: u64) -> u32 {
    x.leading_zeros()
}

// `trailing_zeros` reverses the bits and counts leading zeros; the `rbit`/`clz` pair is adjacent.
// CHECK-LABEL: _tz32:
// CHECK: rbit x9, x9
// CHECK-NEXT: clz x9, x9
#[no_mangle]
pub extern "C" fn tz32(x: u32) -> u32 {
    x.trailing_zeros()
}
