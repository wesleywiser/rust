//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Frames larger than the 12-bit immediate ranges are handled by materializing offsets into a
//! register. The frame size is built in x16 and the stack pointer is adjusted with the
//! extended-register form (`sub sp, sp, x16` / `add sp, sp, x16`) which — unlike the
//! shifted-register form — permits `sp`. A slot beyond the load/store immediate range is reached by
//! forming its address in x16 (`add x16, sp, x16`) and accessing `[x16]`.
#![feature(core_intrinsics)]
#![crate_type = "lib"]

// The 48 KB array forces a frame far larger than 4 KB, so the prologue reserves it with the
// register form rather than a `sub sp, sp, #imm`.
// CHECK-LABEL: _big_frame:
// CHECK: movz x16, #{{[0-9]+}}
// CHECK-NEXT: sub sp, sp, x16
// A slot above the 32 KB load/store immediate range is reached through an x16-formed address.
// CHECK: add x16, sp, x16
// CHECK-NEXT: str x9, [x16, #0]
// The epilogue releases the frame with the matching extended-register add.
// CHECK: add sp, sp, x16
#[no_mangle]
pub extern "C" fn big_frame() -> u64 {
    let mut a = [0u64; 6000];
    a[0] = 7;
    core::hint::black_box(a.as_ptr());
    a[0]
}
