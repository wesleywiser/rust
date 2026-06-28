//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! `core::hint::black_box` is an intrinsic that is marked `must_be_overridden`, so the backend has
//! to handle it directly in `codegen_intrinsic_call`; returning `IntrinsicResult::Fallback` (the
//! default for unimplemented intrinsics) makes `rustc_codegen_ssa` ICE with "intrinsic black_box
//! must be overridden by codegen backend, but isn't". Because this baseline backend performs no
//! optimizations, `black_box` is the identity: the operand passes straight through the stack slots
//! with no optimization-barrier instruction, no `bl` to a fallback helper, and no `brk` abort.

#![crate_type = "lib"]

// The argument is spilled to its slot and immediately read back into the return register; the two
// memory ops are adjacent, proving `black_box` injected nothing between them.
// CHECK-LABEL: _bb_ident:
// CHECK: str x0, [sp, #0]
// CHECK-NEXT: ldr x0, [sp, #0]
// CHECK-NEXT: add sp, sp, #16
// CHECK-NEXT: ldp x29, x30, [sp], #16
// CHECK-NEXT: ret
#[no_mangle]
pub extern "C" fn bb_ident(x: i64) -> i64 {
    core::hint::black_box(x)
}

// `black_box(a)`, `black_box(b)` and `black_box(a + b)` are all identities, so the only arithmetic
// left is the single `add`; the result is spilled and reloaded into `x0` with nothing in between.
// CHECK-LABEL: _bb_sum:
// CHECK: ldr x9, [sp, #0]
// CHECK-NEXT: ldr x10, [sp, #8]
// CHECK-NEXT: add x9, x9, x10
// CHECK-NEXT: str x9, [sp, #16]
// CHECK-NEXT: ldr x0, [sp, #16]
#[no_mangle]
pub extern "C" fn bb_sum(a: i64, b: i64) -> i64 {
    let a = core::hint::black_box(a);
    let b = core::hint::black_box(b);
    core::hint::black_box(a + b)
}
