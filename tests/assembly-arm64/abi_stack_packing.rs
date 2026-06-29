//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Apple AArch64 calling-convention details that differ from the generic AAPCS64, found via
//! `abi-cafe` differential testing against the LLVM backend:
//!
//! 1. Stack-passed arguments are packed at their natural size/alignment, not rounded up to 8-byte
//!    slots — so the 10th `u32` argument (the 2nd stack argument) sits at offset 4, not 8.
//! 2. A 16-byte aggregate is passed in two consecutive GPRs (previously the backend dropped the
//!    high half, treating it as one register).
#![crate_type = "lib"]

#[repr(C)]
#[derive(Copy, Clone)]
pub struct S16 {
    a: u64,
    b: u64,
}

// The callee reads its two stack arguments (arg8, arg9) from the caller's incoming area at
// `fp + 16 + k`. With tight Apple packing the second `u32` stack arg is at k = 4 (`[x29, #20]`),
// not k = 8 — that offset is the regression guard.
// CHECK-LABEL: _read_u32_arg9:
// CHECK: ldr w{{[0-9]+}}, [x29, #16]
// CHECK: ldr w{{[0-9]+}}, [x29, #20]
#[no_mangle]
pub extern "C" fn read_u32_arg9(
    _0: u32, _1: u32, _2: u32, _3: u32, _4: u32, _5: u32, _6: u32, _7: u32, _8: u32, a9: u32,
) -> u32 {
    a9
}

// A caller packs its outgoing stack arguments the same way: the two stack `u32`s go to `[sp, #0]`
// and `[sp, #4]`.
// CHECK-LABEL: _call_ten_u32:
// CHECK: str w{{[0-9]+}}, [sp, #0]
// CHECK: str w{{[0-9]+}}, [sp, #4]
#[no_mangle]
pub unsafe extern "C" fn call_ten_u32(x: u32) {
    unsafe extern "C" {
        fn sink_ten(a: u32, b: u32, c: u32, d: u32, e: u32, f: u32, g: u32, h: u32, i: u32, j: u32);
    }
    sink_ten(x, x, x, x, x, x, x, x, x, x);
}

// A 16-byte aggregate arrives in two consecutive GPRs (x0:x1); both halves are stored. A previous
// bug classified it as a single register, dropping `b` (the high word).
// CHECK-LABEL: _sum_s16:
// CHECK: str x0, [sp
// CHECK: str x1, [sp
#[no_mangle]
pub extern "C" fn sum_s16(s: S16) -> u64 {
    s.a.wrapping_add(s.b)
}
