//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! AAPCS64 `PassMode::Cast` argument/return for multi-register aggregates and HFAs. A non-HFA
//! aggregate of 9..=16 bytes travels in two consecutive integer registers; a homogeneous
//! floating-point aggregate (HFA) of N<=4 members travels in N consecutive SIMD&FP registers. The
//! pre-fix backend collapsed every such cast to a single `i64` register, silently dropping every
//! byte past the first 8 (and using the wrong register bank for HFAs). These checks pin that *all*
//! the registers are read (arguments) and written (returns).
#![crate_type = "lib"]

#[derive(Clone, Copy)] pub struct I16 { pub a: u32, pub b: u32, pub c: u32, pub d: u32 }
#[derive(Clone, Copy)] pub struct D2 { pub x: f64, pub y: f64 }
#[derive(Clone, Copy)] pub struct F4 { pub a: f32, pub b: f32, pub c: f32, pub d: f32 }

// A 16-byte composite arrives in x0:x1 — both must be spilled.
// CHECK-LABEL: _arg16:
// CHECK: str x0, [sp
// CHECK: str x1, [sp
#[no_mangle]
pub extern "C" fn arg16(s: I16) -> u32 {
    s.a ^ s.b ^ s.c ^ s.d
}

// A 2-double HFA arrives in d0:d1 (the SIMD bank), not the integer bank.
// CHECK-LABEL: _argd2:
// CHECK: str d0, [sp
// CHECK: str d1, [sp
#[no_mangle]
pub extern "C" fn argd2(s: D2) -> f64 {
    s.x + s.y
}

// A 4-float HFA arrives in s0:s1:s2:s3.
// CHECK-LABEL: _argf4:
// CHECK: str s0, [sp
// CHECK: str s1, [sp
// CHECK: str s2, [sp
// CHECK: str s3, [sp
#[no_mangle]
pub extern "C" fn argf4(s: F4) -> f32 {
    s.a + s.b + s.c + s.d
}

// A 16-byte composite returns in x0:x1 — both must be written before returning.
// CHECK-LABEL: _ret16:
// CHECK: ldr x0, [sp
// CHECK-NEXT: ldr x1, [sp
// CHECK-NEXT: add sp, sp,
// CHECK-NEXT: ldp x29, x30, [sp], #16
// CHECK-NEXT: ret x30
#[no_mangle]
pub extern "C" fn ret16(x: u32) -> I16 {
    I16 { a: x, b: x, c: x, d: x }
}

// A 2-double HFA returns in d0:d1.
// CHECK-LABEL: _retd2:
// CHECK: ldr d0, [sp
// CHECK-NEXT: ldr d1, [sp
// CHECK-NEXT: add sp, sp,
// CHECK-NEXT: ldp x29, x30, [sp], #16
// CHECK-NEXT: ret x30
#[no_mangle]
pub extern "C" fn retd2(x: f64) -> D2 {
    D2 { x, y: x }
}
