//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! `ptr::read_volatile` of a non-scalar (aggregate) value must copy the whole value into the
//! result place. The pre-fix backend always produced an `OperandValue::Immediate`, which can only
//! hold a single register, so it truncated the aggregate to 8 bytes and left the rest of the
//! destination unwritten — corrupting the surrounding bytes.
//!
//! crossbeam-deque's lock-free work-stealing buffer reads each slot with
//! `ptr::read_volatile::<MaybeUninit<T>>`. For a large `T` this miscompile corrupted every item
//! passing through the deque; via ripgrep's parallel directory walker that flipped a regular
//! file's enum discriminant so it looked like stdin, and the search deadlocked reading stdin.
#![crate_type = "lib"]

pub struct Big {
    a: u64,
    b: u64,
    c: u64,
}

// A 24-byte aggregate read volatile-ly is copied in full (a `memcpy` into the `sret` result
// place), not truncated to a single register store.
// CHECK-LABEL: _read_big:
// CHECK: bl _memcpy
#[no_mangle]
pub unsafe extern "C" fn read_big(p: *const Big) -> Big {
    p.read_volatile()
}
