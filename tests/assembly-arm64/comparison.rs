//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Integer comparisons lower to `cmp` plus a conditional set (`cset`, encoded as `csinc`).
#![crate_type = "lib"]

// CHECK-LABEL: _gt:
// CHECK: cmp x9, x10
// CHECK: csinc
#[no_mangle]
pub extern "C" fn gt(a: u64, b: u64) -> bool {
    a > b
}

// CHECK-LABEL: _eq:
// CHECK: cmp x9, x10
// CHECK: csinc
#[no_mangle]
pub extern "C" fn eq(a: u64, b: u64) -> bool {
    a == b
}
