//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Floating-point comparisons lower to `fcmp` plus a conditional set (`cset`, encoded as `csinc`).
#![crate_type = "lib"]

// CHECK-LABEL: _flt:
// CHECK: fcmp d
// CHECK: csinc w
#[no_mangle]
pub extern "C" fn flt(a: f64, b: f64) -> bool {
    a < b
}

// CHECK-LABEL: _fgt:
// CHECK: fcmp d
// CHECK: csinc w
#[no_mangle]
pub extern "C" fn fgt(a: f64, b: f64) -> bool {
    a > b
}

// CHECK-LABEL: _feq:
// CHECK: fcmp d
// CHECK: csinc w
#[no_mangle]
pub extern "C" fn feq(a: f64, b: f64) -> bool {
    a == b
}

// CHECK-LABEL: _fne:
// CHECK: fcmp d
// CHECK: csinc w
#[no_mangle]
pub extern "C" fn fne(a: f64, b: f64) -> bool {
    a != b
}

// CHECK-LABEL: _fle32:
// CHECK: fcmp s
// CHECK: csinc w
#[no_mangle]
pub extern "C" fn fle32(a: f32, b: f32) -> bool {
    a <= b
}
