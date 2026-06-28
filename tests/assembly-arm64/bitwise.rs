//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Bitwise operations. The first case pins the full sequence; the others pin the operand-loading,
//! the opcode, and the result spill so a wrong register or opcode is caught.
#![crate_type = "lib"]

// CHECK-LABEL: _bit_and:
// CHECK-NEXT: stp x29, x30, [sp, #-16]!
// CHECK-NEXT: mov x29, sp
// CHECK-NEXT: sub sp, sp, #32
// CHECK-NEXT: LBB{{[0-9]+}}_0:
// CHECK-NEXT: str x0, [sp, #0]
// CHECK-NEXT: str x1, [sp, #8]
// CHECK-NEXT: ldr x9, [sp, #0]
// CHECK-NEXT: ldr x10, [sp, #8]
// CHECK-NEXT: and x9, x9, x10
// CHECK-NEXT: str x9, [sp, #16]
// CHECK-NEXT: ldr x0, [sp, #16]
// CHECK-NEXT: add sp, sp, #32
// CHECK-NEXT: ldp x29, x30, [sp], #16
// CHECK-NEXT: ret x30
#[no_mangle]
pub extern "C" fn bit_and(a: u64, b: u64) -> u64 {
    a & b
}

// CHECK-LABEL: _bit_or:
// CHECK: ldr x9, [sp, #0]
// CHECK-NEXT: ldr x10, [sp, #8]
// CHECK-NEXT: orr x9, x9, x10
// CHECK-NEXT: str x9, [sp, #16]
#[no_mangle]
pub extern "C" fn bit_or(a: u64, b: u64) -> u64 {
    a | b
}

// CHECK-LABEL: _bit_xor:
// CHECK: ldr x9, [sp, #0]
// CHECK-NEXT: ldr x10, [sp, #8]
// CHECK-NEXT: eor x9, x9, x10
// CHECK-NEXT: str x9, [sp, #16]
#[no_mangle]
pub extern "C" fn bit_xor(a: u64, b: u64) -> u64 {
    a ^ b
}
