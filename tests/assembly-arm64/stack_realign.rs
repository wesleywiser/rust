//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Locals with an alignment greater than the 16-byte stack alignment require the prologue to
//! dynamically realign `sp` downward (the entry `sp` is only 16-byte aligned). The realignment is
//! `sp &= ~(align - 1)`, computed in x16/x17 (`mov x16, sp; movn x17, #(align-1); and x16, x16, x17;
//! mov sp, x16`) since `sp` can only be written through the SP-aware move form. The epilogue then
//! restores `sp` from the frame pointer (`mov sp, x29`) rather than by adding back a fixed frame
//! size, because the realignment shifted `sp` by a runtime-variable amount.
#![crate_type = "lib"]

#[repr(align(64))]
pub struct A64(u64);

// CHECK-LABEL: _realign:
// CHECK:      mov x29, sp
// The frame is reserved, then sp is masked down to a 64-byte boundary.
// CHECK:      mov x16, sp
// CHECK-NEXT: movn x17, #63
// CHECK-NEXT: and x16, x16, x17
// CHECK-NEXT: mov sp, x16
// The epilogue restores sp from the frame pointer, not via `add sp, sp, #imm`.
// CHECK:      mov sp, x29
// CHECK-NEXT: ldp x29, x30, [sp], #16
// CHECK-NEXT: ret
#[no_mangle]
pub extern "C" fn realign(x: u64) -> u64 {
    let a = A64(x);
    core::hint::black_box(&a);
    a.0
}

// A normally-aligned function must NOT realign: the common prologue/epilogue is unchanged.
// CHECK-LABEL: _no_realign:
// CHECK-NOT:  mov sp, x16
// CHECK:      ret
#[no_mangle]
pub extern "C" fn no_realign(x: u64) -> u64 {
    let a = x;
    core::hint::black_box(&a);
    a
}
