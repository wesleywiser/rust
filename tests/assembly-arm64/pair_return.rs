//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Returning a scalar pair (or a small homogeneous aggregate) by value uses the two-register ABI:
//! field 0 in `x0`/`d0`, field 1 in `x1`/`d1`. The pair is assembled in a stack slot then split
//! into the return registers. Pinning *both* register loads immediately before the epilogue guards
//! against the easy bug of returning only the first field.
#![crate_type = "lib"]

// `swap` returns `(b, a)`: field 0 is `b` (param 1, spilled at offset 8), field 1 is `a` (param 0,
// offset 0). Both source loads and the two-register return are pinned.
// CHECK-LABEL: _swap:
// CHECK: ldr x10, [sp, #8]
// CHECK-NEXT: str x10, [x9, #0]
// CHECK: ldr x10, [sp, #0]
// CHECK-NEXT: str x10, [x9, #0]
// CHECK: ldr x10, [x9, #8]
// CHECK: ldr x0, [sp,
// CHECK-NEXT: ldr x1, [sp,
// CHECK-NEXT: add sp, sp,
// CHECK-NEXT: ldp x29, x30, [sp], #16
// CHECK-NEXT: ret x30
#[no_mangle]
pub extern "C" fn swap(a: u64, b: u64) -> (u64, u64) {
    (b, a)
}

// `overflowing_add` returns `(u64, bool)`, a true scalar pair: the value comes back in `x0` and the
// overflow flag in `x1`. Anchor on the field-1 read (`[x9, #8]`) to skip past the call's own
// argument marshalling, then pin the two-register return.
// CHECK-LABEL: _add_ovf:
// CHECK: ldr x10, [x9, #8]
// CHECK: ldr x0, [sp,
// CHECK-NEXT: ldr x1, [sp,
// CHECK-NEXT: add sp, sp,
// CHECK-NEXT: ldp x29, x30, [sp], #16
// CHECK-NEXT: ret x30
#[no_mangle]
pub extern "C" fn add_ovf(a: u64, b: u64) -> (u64, bool) {
    a.overflowing_add(b)
}

// A homogeneous floating-point pair (an HFA) returns both fields in the FP registers `d0`/`d1`.
// CHECK-LABEL: _fpair:
// CHECK: fadd d16, d16, d17
// CHECK: fsub d16, d16, d17
// CHECK: ldr d16, [x9, #8]
// CHECK: ldr d0, [sp,
// CHECK-NEXT: ldr d1, [sp,
// CHECK-NEXT: add sp, sp,
// CHECK-NEXT: ldp x29, x30, [sp], #16
// CHECK-NEXT: ret x30
#[no_mangle]
pub extern "C" fn fpair(a: f64, b: f64) -> (f64, f64) {
    (a + b, a - b)
}
