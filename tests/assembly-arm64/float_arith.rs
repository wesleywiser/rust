//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Floating-point arithmetic and the FP calling convention: arguments arrive in `d0`/`d1` (or
//! `s0`/`s1` for `f32`), are spilled to the frame, reloaded into scratch FP registers, and the
//! result is returned in `d0`/`s0`.
#![crate_type = "lib"]

// CHECK-LABEL: _fadd64:
// CHECK: str d0, [sp
// CHECK: str d1, [sp
// CHECK: fadd d
// CHECK: ret
#[no_mangle]
pub extern "C" fn fadd64(a: f64, b: f64) -> f64 {
    a + b
}

// CHECK-LABEL: _fsub64:
// CHECK: fsub d
#[no_mangle]
pub extern "C" fn fsub64(a: f64, b: f64) -> f64 {
    a - b
}

// CHECK-LABEL: _fmul64:
// CHECK: fmul d
#[no_mangle]
pub extern "C" fn fmul64(a: f64, b: f64) -> f64 {
    a * b
}

// CHECK-LABEL: _fdiv64:
// CHECK: fdiv d
#[no_mangle]
pub extern "C" fn fdiv64(a: f64, b: f64) -> f64 {
    a / b
}

// CHECK-LABEL: _fneg64:
// CHECK: fneg d
#[no_mangle]
pub extern "C" fn fneg64(a: f64) -> f64 {
    -a
}

// `f32` uses the single-precision (`s`) register views.
// CHECK-LABEL: _fadd32:
// CHECK: str s0, [sp
// CHECK: fadd s
#[no_mangle]
pub extern "C" fn fadd32(a: f32, b: f32) -> f32 {
    a + b
}

// A floating-point constant is materialized into a GPR and moved across with `fmov`.
// CHECK-LABEL: _scale:
// CHECK: fmov d{{[0-9]+}}, x{{[0-9]+}}
// CHECK: fmul d
#[no_mangle]
pub extern "C" fn scale(x: f64) -> f64 {
    x * 2.0
}
