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
    fn utimes(p: *const u8, t: *const u8) -> i32;
}

// CHECK-LABEL: _read_environ:
// CHECK: adrp x{{[0-9]+}}, _environ@GOTPAGE
// CHECK: ldr x{{[0-9]+}}, [x{{[0-9]+}}, _environ@GOTPAGEOFF]
#[no_mangle]
pub fn read_environ() -> *const *const u8 {
    unsafe { environ }
}

// Taking the *address* of a foreign function also needs the GOT (a direct call uses bl, but a fn
// pointer cannot be a direct adrp+add — `_utimes` does not have address). xsv's filetime dep hit this.
// CHECK-LABEL: _utimes_addr:
// CHECK: adrp x{{[0-9]+}}, _utimes@GOTPAGE
// CHECK: ldr x{{[0-9]+}}, [x{{[0-9]+}}, _utimes@GOTPAGEOFF]
#[no_mangle]
pub fn utimes_addr() -> usize {
    utimes as usize
}
