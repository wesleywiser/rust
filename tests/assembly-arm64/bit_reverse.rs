//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! `bswap` (`swap_bytes`) and `bitreverse` (`reverse_bits`) are `must_be_overridden` intrinsics.
//! They lower to AArch64 `rev`/`rbit`, which reverse the whole 32/64-bit register. For a full-width
//! type that is the answer directly; for a sub-word type the reversed low bits land at the top of
//! the register and are shifted back down by `op_bits - N`.

#![crate_type = "lib"]

// Full-width byte swap is a single `rev`.
// CHECK-LABEL: _bswap_u32:
// CHECK: rev w9, w9
// CHECK-NEXT: str w9,
#[no_mangle]
pub extern "C" fn bswap_u32(x: u32) -> u32 {
    x.swap_bytes()
}

// A `u16` byte swap reverses the 32-bit register then shifts the result down by 16.
// CHECK-LABEL: _bswap_u16:
// CHECK: rev w9, w9
// CHECK-NEXT: movz w10, #16
// CHECK-NEXT: lsr w9, w9, w10
#[no_mangle]
pub extern "C" fn bswap_u16(x: u16) -> u16 {
    x.swap_bytes()
}

// A `u8` bit reverse uses `rbit` then shifts down by 24.
// CHECK-LABEL: _brev_u8:
// CHECK: rbit w9, w9
// CHECK-NEXT: movz w10, #24
// CHECK-NEXT: lsr w9, w9, w10
#[no_mangle]
pub extern "C" fn brev_u8(x: u8) -> u8 {
    x.reverse_bits()
}
