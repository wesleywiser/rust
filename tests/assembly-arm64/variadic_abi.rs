//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Apple's AArch64 calling convention passes every variadic argument on the stack, regardless of
//! type — unlike a fixed argument of the same type, which would use the next free GPR/SIMD
//! register. The backend collapsed variadics onto the normal register path, so a call like
//! `snprintf("%d %f", i, f)` put the arguments where the callee never looks for them and printed
//! garbage. These checks pin that a variadic `double`/`long` goes to `[sp]`, while the fixed
//! integer argument stays in `w0`.
#![crate_type = "lib"]
use std::ffi::c_int;

unsafe extern "C" {
    fn vf(n: c_int, ...);
}

// The fixed `n` is in `w0`; the variadic `double` is spilled to the stack, not passed in `d1`.
// CHECK-LABEL: _call_vf:
// CHECK: movz w0, #1
// CHECK: str d{{[0-9]+}}, [sp, #0]
// CHECK: bl _vf
#[no_mangle]
pub fn call_vf(x: f64) {
    unsafe { vf(1, x) }
}

// The fixed `n` is in `w0`; the variadic `long` is spilled to the stack, not passed in `x1`.
// CHECK-LABEL: _call_vi:
// CHECK: movz w0, #1
// CHECK: str x{{[0-9]+}}, [sp, #0]
// CHECK: bl _vf
#[no_mangle]
pub fn call_vi(x: i64) {
    unsafe { vf(1, x) }
}
