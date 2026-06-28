//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Integer comparisons lower to `cmp` plus a conditional set (`cset rd, cc`, encoded as
//! `csinc rd, wzr, wzr, invert(cc)`). The exact (inverted) condition code is pinned for every
//! predicate, signed and unsigned — this is where an inverted-condition or signedness bug hides.
#![crate_type = "lib"]

// The full boolean-producing sequence (cmp, cset, store the `i1`, reload into `w0`).
// CHECK-LABEL: _ueq:
// CHECK: ldr x9, [sp, #0]
// CHECK-NEXT: ldr x10, [sp, #8]
// CHECK-NEXT: cmp x9, x10
// CHECK-NEXT: csinc w9, wzr, wzr, ne
// CHECK-NEXT: strb w9, [sp, #16]
// CHECK-NEXT: ldrb w0, [sp, #16]
#[no_mangle]
pub extern "C" fn ueq(a: u64, b: u64) -> bool {
    a == b
}

// CHECK-LABEL: _une:
// CHECK: cmp x9, x10
// CHECK-NEXT: csinc w9, wzr, wzr, eq
#[no_mangle]
pub extern "C" fn une(a: u64, b: u64) -> bool {
    a != b
}

// CHECK-LABEL: _ugt:
// CHECK: cmp x9, x10
// CHECK-NEXT: csinc w9, wzr, wzr, ls
#[no_mangle]
pub extern "C" fn ugt(a: u64, b: u64) -> bool {
    a > b
}

// CHECK-LABEL: _uge:
// CHECK: cmp x9, x10
// CHECK-NEXT: csinc w9, wzr, wzr, lo
#[no_mangle]
pub extern "C" fn uge(a: u64, b: u64) -> bool {
    a >= b
}

// CHECK-LABEL: _ult:
// CHECK: cmp x9, x10
// CHECK-NEXT: csinc w9, wzr, wzr, hs
#[no_mangle]
pub extern "C" fn ult(a: u64, b: u64) -> bool {
    a < b
}

// CHECK-LABEL: _ule:
// CHECK: cmp x9, x10
// CHECK-NEXT: csinc w9, wzr, wzr, hi
#[no_mangle]
pub extern "C" fn ule(a: u64, b: u64) -> bool {
    a <= b
}

// CHECK-LABEL: _sgt:
// CHECK: cmp x9, x10
// CHECK-NEXT: csinc w9, wzr, wzr, le
#[no_mangle]
pub extern "C" fn sgt(a: i64, b: i64) -> bool {
    a > b
}

// CHECK-LABEL: _sge:
// CHECK: cmp x9, x10
// CHECK-NEXT: csinc w9, wzr, wzr, lt
#[no_mangle]
pub extern "C" fn sge(a: i64, b: i64) -> bool {
    a >= b
}

// CHECK-LABEL: _slt:
// CHECK: cmp x9, x10
// CHECK-NEXT: csinc w9, wzr, wzr, ge
#[no_mangle]
pub extern "C" fn slt(a: i64, b: i64) -> bool {
    a < b
}

// CHECK-LABEL: _sle:
// CHECK: cmp x9, x10
// CHECK-NEXT: csinc w9, wzr, wzr, gt
#[no_mangle]
pub extern "C" fn sle(a: i64, b: i64) -> bool {
    a <= b
}
