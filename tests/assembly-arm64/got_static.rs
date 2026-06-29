//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! A dylib-imported (foreign) static has no in-image address; its address lives in a GOT slot.
//! Referencing it must use `adrp …@GOTPAGE` + `ldr …@GOTPAGEOFF`, not a direct `adrp+add@PAGEOFF`,
//! which ld cannot fix up ("target '_mach_task_self_' does not have address"). rand/tokio's libc
//! signal handling tripped this.
#![crate_type = "lib"]

unsafe extern "C" {
    static environ: *const *const u8;
}

// CHECK-LABEL: _read_environ:
// CHECK: adrp x{{[0-9]+}}, _environ@GOTPAGE
// CHECK: ldr x{{[0-9]+}}, [x{{[0-9]+}}, _environ@GOTPAGEOFF]
#[no_mangle]
pub fn read_environ() -> *const *const u8 {
    unsafe { environ }
}
