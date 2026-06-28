//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Bulk memory operations lower to libc calls with arguments marshalled into `x0` (dst/ptr), `x1`
//! (src, or the fill byte for `memset`), and `x2` (size). Calling the intrinsics directly keeps the
//! call inline so the argument marshalling and the exact callee symbol are pinned in this function
//! (the safe `core::ptr` wrappers would instead emit a call into a separate, non-inlined helper).
#![feature(core_intrinsics)]
#![crate_type = "lib"]

// CHECK-LABEL: _mcopy:
// CHECK: ldr x0, [sp, #0]
// CHECK-NEXT: ldr x1, [sp, #8]
// CHECK-NEXT: ldr x2, [sp, #24]
// CHECK-NEXT: bl _memcpy
#[no_mangle]
pub unsafe extern "C" fn mcopy(dst: *mut u8, src: *const u8, n: usize) {
    core::intrinsics::copy_nonoverlapping(src, dst, n);
}

// CHECK-LABEL: _mmove:
// CHECK: ldr x0, [sp, #0]
// CHECK-NEXT: ldr x1, [sp, #8]
// CHECK-NEXT: ldr x2, [sp, #24]
// CHECK-NEXT: bl _memmove
#[no_mangle]
pub unsafe extern "C" fn mmove(dst: *mut u8, src: *const u8, n: usize) {
    core::intrinsics::copy(src, dst, n);
}

// `memset` takes the fill byte in `w1` (loaded with a byte load), not a full register.
// CHECK-LABEL: _mset:
// CHECK: ldr x0, [sp, #0]
// CHECK-NEXT: ldrb w1, [sp, #8]
// CHECK-NEXT: ldr x2, [sp, #24]
// CHECK-NEXT: bl _memset
#[no_mangle]
pub unsafe extern "C" fn mset(dst: *mut u8, val: u8, n: usize) {
    core::intrinsics::write_bytes(dst, val, n);
}

// `compare_bytes` has `memcmp` semantics; the pointers and length are marshalled into `x0`/`x1`/`x2`
// and the `i32` result comes back in `w0`.
// CHECK-LABEL: _mcmp:
// CHECK: ldr x0, [sp, #0]
// CHECK-NEXT: ldr x1, [sp, #8]
// CHECK-NEXT: ldr x2, [sp, #16]
// CHECK-NEXT: bl _memcmp
#[no_mangle]
pub unsafe extern "C" fn mcmp(a: *const u8, b: *const u8, n: usize) -> i32 {
    core::intrinsics::compare_bytes(a, b, n)
}
