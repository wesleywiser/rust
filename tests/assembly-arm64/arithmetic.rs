//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Integer arithmetic and argument passing. Arguments arrive in `x0`/`x1`, are spilled to the
//! frame, reloaded into the scratch registers `x9`/`x10`, combined, and the result is spilled and
//! reloaded into `x0`. The full sequence is pinned so a wrong register, offset, or opcode is caught.
#![crate_type = "lib"]

// CHECK-LABEL: _add:
// CHECK-NEXT: stp x29, x30, [sp, #-16]!
// CHECK-NEXT: mov x29, sp
// CHECK-NEXT: sub sp, sp, #32
// CHECK-NEXT: LBB{{[0-9]+}}_0:
// CHECK-NEXT: str x0, [sp, #0]
// CHECK-NEXT: str x1, [sp, #8]
// CHECK-NEXT: ldr x9, [sp, #0]
// CHECK-NEXT: ldr x10, [sp, #8]
// CHECK-NEXT: add x9, x9, x10
// CHECK-NEXT: str x9, [sp, #16]
// CHECK-NEXT: ldr x0, [sp, #16]
// CHECK-NEXT: add sp, sp, #32
// CHECK-NEXT: ldp x29, x30, [sp], #16
// CHECK-NEXT: ret x30
#[no_mangle]
pub extern "C" fn add(a: u64, b: u64) -> u64 {
    a + b
}

// CHECK-LABEL: _sub:
// CHECK: ldr x9, [sp, #0]
// CHECK-NEXT: ldr x10, [sp, #8]
// CHECK-NEXT: sub x9, x9, x10
// CHECK-NEXT: str x9, [sp, #16]
#[no_mangle]
pub extern "C" fn sub(a: u64, b: u64) -> u64 {
    a - b
}

// CHECK-LABEL: _mul:
// CHECK: ldr x9, [sp, #0]
// CHECK-NEXT: ldr x10, [sp, #8]
// CHECK-NEXT: mul x9, x9, x10
// CHECK-NEXT: str x9, [sp, #16]
#[no_mangle]
pub extern "C" fn mul(a: u64, b: u64) -> u64 {
    a * b
}

// 32-bit operands use the `w` register views and narrower (`str w`) spills.
// CHECK-LABEL: _add32:
// CHECK-NEXT: stp x29, x30, [sp, #-16]!
// CHECK-NEXT: mov x29, sp
// CHECK-NEXT: sub sp, sp, #16
// CHECK-NEXT: LBB{{[0-9]+}}_0:
// CHECK-NEXT: str w0, [sp, #0]
// CHECK-NEXT: str w1, [sp, #4]
// CHECK-NEXT: ldr w9, [sp, #0]
// CHECK-NEXT: ldr w10, [sp, #4]
// CHECK-NEXT: add w9, w9, w10
// CHECK-NEXT: str w9, [sp, #8]
// CHECK-NEXT: ldr w0, [sp, #8]
// CHECK-NEXT: add sp, sp, #16
// CHECK-NEXT: ldp x29, x30, [sp], #16
// CHECK-NEXT: ret x30
#[no_mangle]
pub extern "C" fn add32(a: u32, b: u32) -> u32 {
    a + b
}
