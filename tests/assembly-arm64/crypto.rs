//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off -Ctarget-feature=+aes,+sha2
//! The ARMv8 crypto-extension intrinsics lower to the real AES/SHA instructions, with the 128-bit
//! vector operands moved between frame slots and `v` registers via `q` loads/stores. Each callee
//! carries `target_feature` so the intrinsic inlines and its exact instruction is pinned. Operands
//! are taken via pointers (`vld1q`/`vst1q`) to keep the focus on the crypto instruction itself.
#![crate_type = "lib"]
use std::arch::aarch64::*;

// AES round: a `q` load of each operand, then `aese`, then a `q` store of the result.
// CHECK-LABEL: _aese_:
// CHECK: ldr q{{[0-9]+}}, [{{.*}}]
// CHECK: aese v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
// CHECK: str q{{[0-9]+}}, [{{.*}}]
#[no_mangle]
#[target_feature(enable = "aes")]
pub unsafe extern "C" fn aese_(d: *const u8, k: *const u8, out: *mut u8) {
    vst1q_u8(out, vaeseq_u8(vld1q_u8(d), vld1q_u8(k)));
}

// CHECK-LABEL: _aesmc_:
// CHECK: aesmc v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
#[no_mangle]
#[target_feature(enable = "aes")]
pub unsafe extern "C" fn aesmc_(d: *const u8, out: *mut u8) {
    vst1q_u8(out, vaesmcq_u8(vld1q_u8(d)));
}

// SHA-256 hash update (three vector operands).
// CHECK-LABEL: _sha256h_:
// CHECK: sha256h q{{[0-9]+}}, q{{[0-9]+}}, v{{[0-9]+}}.4s
#[no_mangle]
#[target_feature(enable = "sha2")]
pub unsafe extern "C" fn sha256h_(a: *const u32, b: *const u32, c: *const u32, out: *mut u32) {
    vst1q_u32(out, vsha256hq_u32(vld1q_u32(a), vld1q_u32(b), vld1q_u32(c)));
}

// SHA-256 schedule (two vector operands).
// CHECK-LABEL: _sha256su0_:
// CHECK: sha256su0 v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
#[no_mangle]
#[target_feature(enable = "sha2")]
pub unsafe extern "C" fn sha256su0_(a: *const u32, b: *const u32, out: *mut u32) {
    vst1q_u32(out, vsha256su0q_u32(vld1q_u32(a), vld1q_u32(b)));
}

// SHA-1 choose round: the `hash_e` operand is a scalar in an `s` register (`fmov`), not a vector.
// CHECK-LABEL: _sha1c_:
// CHECK: fmov s{{[0-9]+}}, w{{[0-9]+}}
// CHECK: sha1c q{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}.4s
#[no_mangle]
#[target_feature(enable = "sha2")]
pub unsafe extern "C" fn sha1c_(a: *const u32, e: u32, c: *const u32, out: *mut u32) {
    vst1q_u32(out, vsha1cq_u32(vld1q_u32(a), e, vld1q_u32(c)));
}

// SHA-1 fixed rotate: scalar 32-bit in and out.
// CHECK-LABEL: _sha1h_:
// CHECK: sha1h s{{[0-9]+}}, s{{[0-9]+}}
#[no_mangle]
#[target_feature(enable = "sha2")]
pub unsafe extern "C" fn sha1h_(e: u32) -> u32 {
    vsha1h_u32(e)
}
