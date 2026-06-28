//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Returning integer constants. Leaf functions need no stack frame beyond the saved FP/LR; wide
//! constants are materialized with a `movz`/`movk` pair.
#![crate_type = "lib"]

// CHECK-LABEL: _five:
// CHECK-NEXT: stp x29, x30, [sp, #-16]!
// CHECK-NEXT: mov x29, sp
// CHECK-NEXT: LBB{{[0-9]+}}_0:
// CHECK-NEXT: movz x0, #5{{$}}
// CHECK-NEXT: ldp x29, x30, [sp], #16
// CHECK-NEXT: ret x30
#[no_mangle]
pub extern "C" fn five() -> u64 {
    5
}

// CHECK-LABEL: _zero:
// CHECK-NEXT: stp x29, x30, [sp, #-16]!
// CHECK-NEXT: mov x29, sp
// CHECK-NEXT: LBB{{[0-9]+}}_0:
// CHECK-NEXT: movz x0, #0{{$}}
// CHECK-NEXT: ldp x29, x30, [sp], #16
// CHECK-NEXT: ret x30
#[no_mangle]
pub extern "C" fn zero() -> u64 {
    0
}

// A constant wider than 16 bits needs a `movz` (low half) plus a shifted `movk` (bits 16..31).
// CHECK-LABEL: _wide:
// CHECK-NEXT: stp x29, x30, [sp, #-16]!
// CHECK-NEXT: mov x29, sp
// CHECK-NEXT: LBB{{[0-9]+}}_0:
// CHECK-NEXT: movz x0, #22136{{$}}
// CHECK-NEXT: movk x0, #4660, lsl #16
// CHECK-NEXT: ldp x29, x30, [sp], #16
// CHECK-NEXT: ret x30
#[no_mangle]
pub extern "C" fn wide() -> u64 {
    0x1234_5678
}
