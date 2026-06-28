//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Control flow: a branch (`SwitchInt` on a `bool`) lowers to a compare-and-branch (`cbnz`) to the
//! `then` block plus an unconditional branch to the `else` block, with one block per arm joining at
//! a merge block. The block targets and per-arm values are pinned (`[[FN]]` captures the function's
//! label index) so a swapped arm or wrong branch target is caught.
#![crate_type = "lib"]

// CHECK-LABEL: _max:
// CHECK: cmp x9, x10
// CHECK-NEXT: csinc w9, wzr, wzr, ls
// CHECK-NEXT: strb w9, [sp, #32]
// CHECK-NEXT: ldrb w9, [sp, #32]
// CHECK-NEXT: cbnz w9, LBB[[FN:[0-9]+]]_2
// CHECK-NEXT: b LBB[[FN]]_1
// The `else` arm returns `b` (spilled at offset 8).
// CHECK: LBB[[FN]]_1:
// CHECK-NEXT: ldr x9, [sp, #24]
// CHECK-NEXT: ldr x10, [sp, #8]
// CHECK-NEXT: str x10, [x9, #0]
// CHECK-NEXT: b LBB[[FN]]_3
// The `then` arm returns `a` (spilled at offset 0).
// CHECK: LBB[[FN]]_2:
// CHECK-NEXT: ldr x9, [sp, #24]
// CHECK-NEXT: ldr x10, [sp, #0]
// CHECK-NEXT: str x10, [x9, #0]
// CHECK-NEXT: b LBB[[FN]]_3
// CHECK: LBB[[FN]]_3:
// CHECK: ret x30
#[no_mangle]
pub extern "C" fn max(a: u64, b: u64) -> u64 {
    if a > b { a } else { b }
}
