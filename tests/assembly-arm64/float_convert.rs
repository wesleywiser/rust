//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Conversions between integers and floating-point, and between float widths.
//!
//! - integer -> float: `scvtf`/`ucvtf` (the source is first widened to 64 bits)
//! - float -> integer: `fcvtzs`/`fcvtzu` (round toward zero, saturating)
//! - float width: `fcvt`
#![crate_type = "lib"]

// CHECK-LABEL: _i2d:
// CHECK: scvtf d
// CHECK: ret
#[no_mangle]
pub extern "C" fn i2d(x: i64) -> f64 {
    x as f64
}

// CHECK-LABEL: _u2d:
// CHECK: ucvtf d
#[no_mangle]
pub extern "C" fn u2d(x: u64) -> f64 {
    x as f64
}

// CHECK-LABEL: _i2f:
// CHECK: scvtf s
#[no_mangle]
pub extern "C" fn i2f(x: i32) -> f32 {
    x as f32
}

// CHECK-LABEL: _d2i:
// CHECK: fcvtzs x
#[no_mangle]
pub extern "C" fn d2i(x: f64) -> i64 {
    x as i64
}

// CHECK-LABEL: _d2u:
// CHECK: fcvtzu x
#[no_mangle]
pub extern "C" fn d2u(x: f64) -> u64 {
    x as u64
}

// CHECK-LABEL: _f2d:
// CHECK: fcvt d
#[no_mangle]
pub extern "C" fn f2d(x: f32) -> f64 {
    x as f64
}

// CHECK-LABEL: _d2f:
// CHECK: fcvt s
#[no_mangle]
pub extern "C" fn d2f(x: f64) -> f32 {
    x as f32
}
