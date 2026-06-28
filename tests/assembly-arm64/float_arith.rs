//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Floating-point arithmetic and the FP calling convention: arguments arrive in `d0`/`d1` (or
//! `s0`/`s1` for `f32`), are spilled to the frame, reloaded into scratch FP registers `d16`/`d17`,
//! combined, and returned in `d0`/`s0`. Full sequences are pinned to catch a wrong FP register,
//! width (`d` vs `s`), or opcode.
#![crate_type = "lib"]

// CHECK-LABEL: _fadd64:
// CHECK-NEXT: stp x29, x30, [sp, #-16]!
// CHECK-NEXT: mov x29, sp
// CHECK-NEXT: sub sp, sp, #32
// CHECK-NEXT: LBB{{[0-9]+}}_0:
// CHECK-NEXT: str d0, [sp, #0]
// CHECK-NEXT: str d1, [sp, #8]
// CHECK-NEXT: ldr d16, [sp, #0]
// CHECK-NEXT: ldr d17, [sp, #8]
// CHECK-NEXT: fadd d16, d16, d17
// CHECK-NEXT: str d16, [sp, #16]
// CHECK-NEXT: ldr d0, [sp, #16]
// CHECK-NEXT: add sp, sp, #32
// CHECK-NEXT: ldp x29, x30, [sp], #16
// CHECK-NEXT: ret x30
#[no_mangle]
pub extern "C" fn fadd64(a: f64, b: f64) -> f64 {
    a + b
}

// CHECK-LABEL: _fsub64:
// CHECK: ldr d16, [sp, #0]
// CHECK-NEXT: ldr d17, [sp, #8]
// CHECK-NEXT: fsub d16, d16, d17
// CHECK-NEXT: str d16, [sp, #16]
#[no_mangle]
pub extern "C" fn fsub64(a: f64, b: f64) -> f64 {
    a - b
}

// CHECK-LABEL: _fmul64:
// CHECK: ldr d16, [sp, #0]
// CHECK-NEXT: ldr d17, [sp, #8]
// CHECK-NEXT: fmul d16, d16, d17
// CHECK-NEXT: str d16, [sp, #16]
#[no_mangle]
pub extern "C" fn fmul64(a: f64, b: f64) -> f64 {
    a * b
}

// CHECK-LABEL: _fdiv64:
// CHECK: ldr d16, [sp, #0]
// CHECK-NEXT: ldr d17, [sp, #8]
// CHECK-NEXT: fdiv d16, d16, d17
// CHECK-NEXT: str d16, [sp, #16]
#[no_mangle]
pub extern "C" fn fdiv64(a: f64, b: f64) -> f64 {
    a / b
}

// Unary negation is a single-source `fneg` (no second operand loaded).
// CHECK-LABEL: _fneg64:
// CHECK: str d0, [sp, #0]
// CHECK-NEXT: ldr d16, [sp, #0]
// CHECK-NEXT: fneg d16, d16
// CHECK-NEXT: str d16, [sp, #8]
// CHECK-NEXT: ldr d0, [sp, #8]
#[no_mangle]
pub extern "C" fn fneg64(a: f64) -> f64 {
    -a
}

// `f32` uses the single-precision (`s`) register views and 4-byte spills.
// CHECK-LABEL: _fadd32:
// CHECK: str s0, [sp, #0]
// CHECK-NEXT: str s1, [sp, #4]
// CHECK-NEXT: ldr s16, [sp, #0]
// CHECK-NEXT: ldr s17, [sp, #4]
// CHECK-NEXT: fadd s16, s16, s17
// CHECK-NEXT: str s16, [sp, #8]
// CHECK-NEXT: ldr s0, [sp, #8]
#[no_mangle]
pub extern "C" fn fadd32(a: f32, b: f32) -> f32 {
    a + b
}

// A floating-point constant is materialized into a GPR (`2.0` = 0x4000_0000_0000_0000) and moved
// across with `fmov` before the multiply.
// CHECK-LABEL: _scale:
// CHECK: ldr d16, [sp, #0]
// CHECK-NEXT: movz x9, #0{{$}}
// CHECK-NEXT: movk x9, #16384, lsl #48
// CHECK-NEXT: fmov d17, x9
// CHECK-NEXT: fmul d16, d16, d17
#[no_mangle]
pub extern "C" fn scale(x: f64) -> f64 {
    x * 2.0
}
