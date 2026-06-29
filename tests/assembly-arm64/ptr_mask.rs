//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! `ptr_mask(ptr, mask)` masks a pointer's address bits — a plain 64-bit AND. The backend
//! previously had no override, so codegen aborted with "intrinsic ptr_mask must be overridden".
#![crate_type = "lib"]
#![feature(core_intrinsics)]

// CHECK-LABEL: _align_down:
// CHECK: and x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: ret
#[no_mangle]
pub fn align_down(p: *const u8) -> *const u8 {
    core::intrinsics::ptr_mask(p, !0xf)
}
