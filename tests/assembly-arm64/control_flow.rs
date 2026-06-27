//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Control flow: a branch (`SwitchInt` on a `bool`) lowers to a compare-and-branch plus an
//! unconditional branch, with one block per arm.
#![crate_type = "lib"]

// CHECK-LABEL: _max:
// CHECK: cmp x9, x10
// CHECK: csinc
// CHECK: cbnz
// CHECK: ret
#[no_mangle]
pub extern "C" fn max(a: u64, b: u64) -> u64 {
    if a > b { a } else { b }
}
