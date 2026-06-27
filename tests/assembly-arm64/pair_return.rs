//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Returning a scalar pair (or a small homogeneous aggregate) by value uses the two-register ABI:
//! field 0 in `x0`/`d0`, field 1 in `x1`/`d1`. The pair is assembled in a stack slot then split
//! into the return registers. The two-register loads guard against only returning the first field.
#![crate_type = "lib"]

// CHECK-LABEL: _swap:
// CHECK: ldr x0,
// CHECK: ldr x1,
// CHECK: ret
#[no_mangle]
pub extern "C" fn swap(a: u64, b: u64) -> (u64, u64) {
    (b, a)
}

// `overflowing_add` returns `(u64, bool)` (a true scalar pair); both fields flow through.
// CHECK-LABEL: _add_ovf:
// CHECK: adds
// CHECK: ldr x0,
// CHECK: ret
#[no_mangle]
pub extern "C" fn add_ovf(a: u64, b: u64) -> (u64, bool) {
    a.overflowing_add(b)
}

// A homogeneous floating-point pair returns both fields in the FP registers `d0`/`d1`.
// CHECK-LABEL: _fpair:
// CHECK: fadd d
// CHECK: fsub d
// CHECK: ldr d0,
// CHECK: ldr d1,
// CHECK: ret
#[no_mangle]
pub extern "C" fn fpair(a: f64, b: f64) -> (f64, f64) {
    (a + b, a - b)
}
