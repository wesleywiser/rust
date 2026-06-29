//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off -Ctarget-feature=+crc
//! The dedicated AArch64 CRC32 instructions (`crc32{c}{b,h,w,x}`) lower the `__crc32*` intrinsics
//! used by checksum crates such as `crc32fast`. The CRC accumulator stays a 32-bit register; only
//! the data operand widens to an `x` register for the 64-bit (`d`/`cd`) forms. Each callee carries
//! `target_feature(crc)` so the intrinsic inlines and its exact instruction is pinned.
#![crate_type = "lib"]

use std::arch::aarch64::{__crc32b, __crc32cb, __crc32cd, __crc32d};

// 8-bit data -> crc32b, all-`w` operands.
// CHECK-LABEL: _crc_b:
// CHECK: crc32b w{{[0-9]+}}, w{{[0-9]+}}, w{{[0-9]+}}
#[no_mangle]
#[target_feature(enable = "crc")]
pub unsafe extern "C" fn crc_b(crc: u32, data: u8) -> u32 {
    __crc32b(crc, data)
}

// 64-bit data -> crc32x with an `x` data operand (accumulator still `w`).
// CHECK-LABEL: _crc_x:
// CHECK: crc32x w{{[0-9]+}}, w{{[0-9]+}}, x{{[0-9]+}}
#[no_mangle]
#[target_feature(enable = "crc")]
pub unsafe extern "C" fn crc_x(crc: u32, data: u64) -> u32 {
    __crc32d(crc, data)
}

// Castagnoli (CRC32C) byte form.
// CHECK-LABEL: _crc_cb:
// CHECK: crc32cb w{{[0-9]+}}, w{{[0-9]+}}, w{{[0-9]+}}
#[no_mangle]
#[target_feature(enable = "crc")]
pub unsafe extern "C" fn crc_cb(crc: u32, data: u8) -> u32 {
    __crc32cb(crc, data)
}

// Castagnoli 64-bit form.
// CHECK-LABEL: _crc_cx:
// CHECK: crc32cx w{{[0-9]+}}, w{{[0-9]+}}, x{{[0-9]+}}
#[no_mangle]
#[target_feature(enable = "crc")]
pub unsafe extern "C" fn crc_cx(crc: u32, data: u64) -> u32 {
    __crc32cd(crc, data)
}
