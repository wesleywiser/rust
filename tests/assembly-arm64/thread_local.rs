//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Thread-local statics use the macOS thread-local-variable (TLV) model. Accessing one loads the
//! address of its descriptor (via the `@TLVPPAGE`/`@TLVPPAGEOFF` relocations) and calls the thunk
//! stored in the descriptor's first word; the thunk returns the per-thread address of the variable.
//! The static itself is emitted as a `$tlv$init` initializer in `__thread_data` plus a three-word
//! descriptor in `__thread_vars` whose first word references the dyld-provided `__tlv_bootstrap`.
#![feature(thread_local)]
#![crate_type = "lib"]

// CHECK-LABEL: _get_count:
// CHECK: adrp x0, [[VAR:[0-9A-Za-z_]+]]@TLVPPAGE
// CHECK-NEXT: ldr x0, [x0, [[VAR]]@TLVPPAGEOFF]
// CHECK-NEXT: ldr x8, [x0, #0]
// CHECK-NEXT: blr x8
#[thread_local]
pub static mut COUNT: u64 = 0;

#[no_mangle]
pub fn get_count() -> u64 {
    unsafe { COUNT }
}

// The descriptor lives in `__thread_vars` and references the dyld bootstrap thunk.
// CHECK: .section __DATA,__thread_vars
// CHECK: .quad __tlv_bootstrap
