//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! `f16` uses the native half-precision instructions (`fadd h`, `fcvt`, ...) on Apple Silicon's
//! FEAT_FP16. `f128` (IEEE binary128, no AArch64 hardware) routes every operation through
//! `compiler_builtins` libcalls over 16-byte values held in `q` registers.
#![crate_type = "lib"]
#![feature(f16, f128)]

// CHECK-LABEL: _h_add:
// CHECK: fadd h{{[0-9]+}}, h{{[0-9]+}}, h{{[0-9]+}}
#[no_mangle]
pub extern "C" fn h_add(a: f16, b: f16) -> f16 {
    a + b
}

// CHECK-LABEL: _h_mul:
// CHECK: fmul h{{[0-9]+}}, h{{[0-9]+}}, h{{[0-9]+}}
#[no_mangle]
pub extern "C" fn h_mul(a: f16, b: f16) -> f16 {
    a * b
}

// f16 -> f32 widening is a single `fcvt`.
// CHECK-LABEL: _h_to_f32:
// CHECK: fcvt s{{[0-9]+}}, h{{[0-9]+}}
#[no_mangle]
pub extern "C" fn h_to_f32(a: f16) -> f32 {
    a as f32
}

// f128 arithmetic / comparison / conversions are all compiler-builtins libcalls.
// CHECK-LABEL: _q_add:
// CHECK: bl ___addtf3
#[no_mangle]
pub extern "C" fn q_add(a: f128, b: f128) -> f128 {
    a + b
}

// CHECK-LABEL: _q_div:
// CHECK: bl ___divtf3
#[no_mangle]
pub extern "C" fn q_div(a: f128, b: f128) -> f128 {
    a / b
}

// CHECK-LABEL: _q_lt:
// CHECK: bl ___lttf2
#[no_mangle]
pub extern "C" fn q_lt(a: f128, b: f128) -> bool {
    a < b
}

// CHECK-LABEL: _q_to_f64:
// CHECK: bl ___trunctfdf2
#[no_mangle]
pub extern "C" fn q_to_f64(a: f128) -> f64 {
    a as f64
}

// CHECK-LABEL: _f64_to_q:
// CHECK: bl ___extenddftf2
#[no_mangle]
pub extern "C" fn f64_to_q(a: f64) -> f128 {
    a as f128
}
