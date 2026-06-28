//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Floating-point comparisons lower to `fcmp` plus a conditional set (`cset cc`, encoded as
//! `csinc rd, wzr, wzr, invert(cc)`). The exact inverted condition encodes the ordered/unordered
//! (NaN) semantics — e.g. `a < b` is the *ordered* less-than `mi`, false for NaN — so each is pinned.
#![crate_type = "lib"]

// The full sequence: load both operands into the scratch FP registers, compare, set the `i1`.
// CHECK-LABEL: _feq:
// CHECK: ldr d16, [sp, #0]
// CHECK-NEXT: ldr d17, [sp, #8]
// CHECK-NEXT: fcmp d16, d17
// CHECK-NEXT: csinc w9, wzr, wzr, ne
// CHECK-NEXT: strb w9, [sp, #16]
#[no_mangle]
pub extern "C" fn feq(a: f64, b: f64) -> bool {
    a == b
}

// CHECK-LABEL: _fne:
// CHECK: fcmp d16, d17
// CHECK-NEXT: csinc w9, wzr, wzr, eq
#[no_mangle]
pub extern "C" fn fne(a: f64, b: f64) -> bool {
    a != b
}

// `a < b` is ordered-less-than: `mi` (false when unordered, i.e. a NaN operand).
// CHECK-LABEL: _flt:
// CHECK: fcmp d16, d17
// CHECK-NEXT: csinc w9, wzr, wzr, pl
#[no_mangle]
pub extern "C" fn flt(a: f64, b: f64) -> bool {
    a < b
}

// CHECK-LABEL: _fle:
// CHECK: fcmp d16, d17
// CHECK-NEXT: csinc w9, wzr, wzr, hi
#[no_mangle]
pub extern "C" fn fle(a: f64, b: f64) -> bool {
    a <= b
}

// CHECK-LABEL: _fgt:
// CHECK: fcmp d16, d17
// CHECK-NEXT: csinc w9, wzr, wzr, le
#[no_mangle]
pub extern "C" fn fgt(a: f64, b: f64) -> bool {
    a > b
}

// CHECK-LABEL: _fge:
// CHECK: fcmp d16, d17
// CHECK-NEXT: csinc w9, wzr, wzr, lt
#[no_mangle]
pub extern "C" fn fge(a: f64, b: f64) -> bool {
    a >= b
}

// `f32` compares use the single-precision register views.
// CHECK-LABEL: _fle32:
// CHECK: fcmp s16, s17
// CHECK-NEXT: csinc w9, wzr, wzr, hi
#[no_mangle]
pub extern "C" fn fle32(a: f32, b: f32) -> bool {
    a <= b
}
