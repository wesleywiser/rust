//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Rust's `float as int` casts saturate. AArch64's `fcvtzu`/`fcvtzs` already saturate to the 32/64-
//! bit register and map NaN to 0, but for a narrower destination (`u8`/`i8`/`u16`/`i16`) the result
//! must be clamped again into that type's range — otherwise `300.0 as u8` would truncate to `44`
//! instead of saturating to `255`. Destinations that already span the operation width get no clamp.

#![crate_type = "lib"]

// `f32 as u8`: convert, then clamp to the `u8` maximum (255) with a conditional select.
// CHECK-LABEL: _f2u8:
// CHECK: fcvtzu w9, s16
// CHECK: movz w10, #255
// CHECK: cmp w9, w10
// CHECK: csel
#[no_mangle]
pub extern "C" fn f2u8(x: f32) -> u8 {
    x as u8
}

// `f32 as u32`: the destination already spans the operation width, so the convert is used directly
// with no clamping sequence — the spill follows immediately.
// CHECK-LABEL: _f2u32:
// CHECK: fcvtzu w9, s16
// CHECK-NEXT: str w9,
#[no_mangle]
pub extern "C" fn f2u32(x: f32) -> u32 {
    x as u32
}
