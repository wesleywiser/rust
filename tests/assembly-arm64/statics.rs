//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! Statics and constant data. Reading a static forms its address with `adrp`/`add` (a
//! `@PAGE`/`@PAGEOFF` pair) then loads through it. Initializers are emitted as little-endian data
//! under the static's symbol, and a pointer field becomes a `.quad <symbol>` relocation.
#![crate_type = "lib"]

// Taking a static's address forms the `@PAGE`/`@PAGEOFF` pair. (A *read* of an immutable static
// would be const-folded under optimization, but its address cannot be.)
// CHECK-LABEL: _value_addr:
// CHECK: adrp x0, _VALUE@PAGE
// CHECK-NEXT: add x0, x0, _VALUE@PAGEOFF
#[no_mangle]
pub extern "C" fn value_addr() -> *const u64 {
    &raw const VALUE
}

#[no_mangle]
pub static VALUE: u64 = 42;

static INNER: i32 = 7;

#[no_mangle]
pub static PTR: &i32 = &INNER;

#[no_mangle]
pub static ARR: [u32; 4] = [10, 20, 30, 40];

// The `u64` initializer is emitted as 8 little-endian bytes (42 = 0x2a).
// CHECK-LABEL: _VALUE:
// CHECK-NEXT: .byte 0x2a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00

// The pointer static lowers to a relocation to the pointee's symbol.
// CHECK-LABEL: _PTR:
// CHECK-NEXT: .quad {{.*}}INNER

// The array initializer is emitted as little-endian element bytes (10, 20, 30, 40).
// CHECK-LABEL: _ARR:
// CHECK-NEXT: .byte 0x0a, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x28, 0x00, 0x00, 0x00
