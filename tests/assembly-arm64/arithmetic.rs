//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Integer arithmetic and argument passing (arguments arrive in `x0`/`x1` and are spilled to the
//! frame, then reloaded into scratch registers for each operation).
#![crate_type = "lib"]

// CHECK-LABEL: _add:
// CHECK: str x0, [sp
// CHECK: str x1, [sp
// CHECK: add x9, x9, x10
// CHECK: ret
#[no_mangle]
pub extern "C" fn add(a: u64, b: u64) -> u64 {
    a + b
}

// CHECK-LABEL: _sub:
// CHECK: sub x9, x9, x10
#[no_mangle]
pub extern "C" fn sub(a: u64, b: u64) -> u64 {
    a - b
}

// CHECK-LABEL: _mul:
// CHECK: mul x9, x9, x10
#[no_mangle]
pub extern "C" fn mul(a: u64, b: u64) -> u64 {
    a * b
}
