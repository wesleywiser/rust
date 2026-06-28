//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! A small aggregate passed by value uses `PassMode::Cast`: it travels in a register of the cast
//! type, so it must be spilled and stored at the cast's true width. A 4-byte enum casts to an
//! `i32`; typing the parameter as a full `i64` and storing 8 bytes would overflow the 4-byte
//! aggregate and clobber the adjacent stack slot. That exact bug corrupted a pointer in
//! regex-automata's DFA construction (`set_transition`) and crashed ripgrep with a wild `memmove`.
#![crate_type = "lib"]

pub enum E {
    A(u8),
    B(u16),
}

// The 4-byte Cast parameter is spilled with a 4-byte (`w`) store. The pre-fix codegen typed it as
// an `i64` and used an 8-byte `str x0` here, which overflowed the aggregate into the next slot.
// CHECK-LABEL: _pass_on:
// CHECK: str w0, [sp
#[no_mangle]
pub extern "C" fn pass_on(e: E) -> usize {
    inner(e)
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn inner(e: E) -> usize {
    match e {
        E::A(b) => b as usize,
        E::B(b) => b as usize,
    }
}
