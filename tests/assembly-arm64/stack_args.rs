//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Arguments beyond the eight integer (`x0..x7`) or eight floating-point (`d0..d7`) register
//! slots are passed on the stack. A frame-layout pre-pass scans the function's calls before any
//! code is emitted and reserves an outgoing-argument area at the very bottom of the frame, so a
//! caller writes spilled arguments to `[sp, #k]` while local slots are placed *above* that area
//! and never alias it. A callee reads its own incoming stack arguments from just above the saved
//! frame record: the frame pointer addresses the saved `fp`/`lr` pair, so the first stack
//! argument lives at `[x29, #16]`.
#![crate_type = "lib"]

// The ninth and tenth integer arguments are read from the incoming stack area at `[x29, #16]`
// and `[x29, #24]`.
// CHECK-LABEL: _int_callee:
// CHECK: ldr x9, [x29, #16]
// CHECK: ldr x9, [x29, #24]
#[no_mangle]
pub extern "C" fn int_callee(
    a: i64,
    b: i64,
    c: i64,
    d: i64,
    e: i64,
    f: i64,
    g: i64,
    h: i64,
    i: i64,
    j: i64,
) -> i64 {
    i + j
}

// The first eight integer arguments go in `x0..x7`; the ninth and tenth are written to the
// outgoing-argument area at `[sp, #0]` and `[sp, #8]`. The returned value is then spilled to a
// local slot at `[sp, #16]`, confirming locals sit above the 16-byte outgoing area.
// CHECK-LABEL: _int_caller:
// CHECK: str x9, [sp, #0]
// CHECK: str x9, [sp, #8]
// CHECK: bl _int_callee
// CHECK: str x0, [sp, #16]
#[no_mangle]
pub extern "C" fn int_caller() -> i64 {
    int_callee(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
}

// Floating-point arguments use a separate bank (`d0..d7`); the ninth and tenth `f64` arguments
// are read from the incoming stack area with `ldr d16, [x29, #16]` / `[x29, #24]`.
// CHECK-LABEL: _float_callee:
// CHECK: ldr d16, [x29, #16]
// CHECK: ldr d16, [x29, #24]
#[no_mangle]
pub extern "C" fn float_callee(
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    g: f64,
    h: f64,
    i: f64,
    j: f64,
) -> f64 {
    i + j
}

// A caller passing ten `f64` arguments fills `d0..d7`, then writes the spilled pair into the
// outgoing area with `str d16, [sp, #0]` / `[sp, #8]`.
// CHECK-LABEL: _float_caller:
// CHECK: str d16, [sp, #0]
// CHECK: str d16, [sp, #8]
// CHECK: bl _float_callee
#[no_mangle]
pub extern "C" fn float_caller(x: f64) -> f64 {
    float_callee(x, x, x, x, x, x, x, x, x, x)
}
