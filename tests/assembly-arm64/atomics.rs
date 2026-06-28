//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Atomics lower to LSE single-instruction atomics: load-acquire/store-release (`ldar`/`stlr`), the
//! `LD<op>`/`swp` read-modify-write family, and `cas` for compare-exchange. The acquire/release
//! ordering picks the suffix (`a`/`l`/`al`); relaxed atomics are plain loads/stores. AArch64 has no
//! atomic subtract or and, so those negate/complement the operand and use `ldadd`/`ldclr`. The
//! intrinsics are called directly so each atomic stays inline and its exact instruction is pinned.
#![feature(core_intrinsics)]
#![crate_type = "lib"]

use std::intrinsics::{
    AtomicOrdering, atomic_and, atomic_cxchg, atomic_load, atomic_or, atomic_store, atomic_xadd,
    atomic_xchg, atomic_xsub,
};

// SeqCst load -> load-acquire.
// CHECK-LABEL: _a_load:
// CHECK: ldar x10, [x9]
#[no_mangle]
pub unsafe extern "C" fn a_load(p: *mut u64) -> u64 {
    atomic_load::<u64, { AtomicOrdering::SeqCst }>(p)
}

// Relaxed load -> plain load, no barrier.
// CHECK-LABEL: _a_load_relaxed:
// CHECK: ldr x10, [x9, #0]
#[no_mangle]
pub unsafe extern "C" fn a_load_relaxed(p: *mut u64) -> u64 {
    atomic_load::<u64, { AtomicOrdering::Relaxed }>(p)
}

// SeqCst store -> store-release.
// CHECK-LABEL: _a_store:
// CHECK: stlr x10, [x9]
#[no_mangle]
pub unsafe extern "C" fn a_store(p: *mut u64, v: u64) {
    atomic_store::<u64, { AtomicOrdering::SeqCst }>(p, v)
}

// fetch_add (acq+rel) -> ldaddal; the old value comes back in the destination (x11).
// CHECK-LABEL: _a_add:
// CHECK: ldaddal x10, x11, [x9]
#[no_mangle]
pub unsafe extern "C" fn a_add(p: *mut u64, v: u64) -> u64 {
    atomic_xadd::<u64, u64, { AtomicOrdering::SeqCst }>(p, v)
}

// fetch_sub has no LSE op: negate the operand (`sub x9, xzr-form`) then ldadd.
// CHECK-LABEL: _a_sub:
// CHECK: sub x9, x9, x10
// CHECK: ldaddal x10, x11, [x9]
#[no_mangle]
pub unsafe extern "C" fn a_sub(p: *mut u64, v: u64) -> u64 {
    atomic_xsub::<u64, u64, { AtomicOrdering::SeqCst }>(p, v)
}

// fetch_and has no LSE op: complement the operand (`eor` with all-ones) then ldclr.
// CHECK-LABEL: _a_and:
// CHECK: eor x9, x9, x10
// CHECK: ldclral x10, x11, [x9]
#[no_mangle]
pub unsafe extern "C" fn a_and(p: *mut u64, v: u64) -> u64 {
    atomic_and::<u64, u64, { AtomicOrdering::SeqCst }>(p, v)
}

// Relaxed fetch_or on a `u32` -> ldset on the `w` registers, no ordering suffix.
// CHECK-LABEL: _a_or:
// CHECK: ldset w10, w11, [x9]
#[no_mangle]
pub unsafe extern "C" fn a_or(p: *mut u32, v: u32) -> u32 {
    atomic_or::<u32, u32, { AtomicOrdering::Relaxed }>(p, v)
}

// Acquire swap -> swpa.
// CHECK-LABEL: _a_xchg:
// CHECK: swpa x10, x11, [x9]
#[no_mangle]
pub unsafe extern "C" fn a_xchg(p: *mut u64, v: u64) -> u64 {
    atomic_xchg::<u64, { AtomicOrdering::Acquire }>(p, v)
}

// compare_exchange -> casal, then compare the returned old value against the comparand to produce
// the success flag.
// CHECK-LABEL: _a_cas:
// CHECK: casal x10, x11, [x9]
// CHECK-NEXT: cmp x10, x12
// CHECK-NEXT: csinc w13, wzr, wzr, ne
#[no_mangle]
pub unsafe extern "C" fn a_cas(p: *mut u64, old: u64, new: u64) -> bool {
    atomic_cxchg::<u64, { AtomicOrdering::SeqCst }, { AtomicOrdering::SeqCst }>(p, old, new).1
}
