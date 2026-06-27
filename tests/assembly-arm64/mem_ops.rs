//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Bulk memory operations lower to libc calls (`memcpy`/`memmove`/`memset`) with arguments in
//! `x0`/`x1`/`x2`.
#![crate_type = "lib"]

// CHECK-LABEL: _do_copy:
// CHECK: bl _memcpy
#[no_mangle]
pub unsafe extern "C" fn do_copy(dst: *mut u8, src: *const u8) {
    core::ptr::copy_nonoverlapping(src, dst, 64);
}

// CHECK-LABEL: _do_move:
// CHECK: bl _memmove
#[no_mangle]
pub unsafe extern "C" fn do_move(dst: *mut u8, src: *const u8) {
    core::ptr::copy(src, dst, 64);
}

// CHECK-LABEL: _do_fill:
// CHECK: bl _memset
#[no_mangle]
pub unsafe extern "C" fn do_fill(dst: *mut u8, val: u8) {
    core::ptr::write_bytes(dst, val, 64);
}
