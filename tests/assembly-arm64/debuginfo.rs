//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -g -Coverflow-checks=off
//! With `-g`, the backend threads source locations through the instruction stream as `.loc`
//! markers (`Inst::DebugLoc`). The textual asm path renders them as comments; the binary path turns
//! them into a `.debug_line` line-number program and per-function `DW_TAG_subprogram` DIEs (so lldb
//! recovers function names, file/line locations, and line breakpoints). This test guards the
//! line-tracking path by asserting the markers are emitted within each function body.
#![crate_type = "lib"]

// Each statement is preceded by a `.loc file<n> <line> <col>` marker. A two-statement function
// therefore emits at least two markers after its label.
// CHECK-LABEL: _dbg_add:
// CHECK: // .loc file{{[0-9]+ [0-9]+ [0-9]+}}
// CHECK: // .loc file{{[0-9]+ [0-9]+ [0-9]+}}
#[no_mangle]
pub extern "C" fn dbg_add(a: i32, b: i32) -> i32 {
    let c = a + b;
    c * 2
}

// A second function gets its own markers, confirming locations are tracked per function.
// CHECK-LABEL: _dbg_double:
// CHECK: // .loc file{{[0-9]+ [0-9]+ [0-9]+}}
#[no_mangle]
pub extern "C" fn dbg_double(x: i64) -> i64 {
    x + x
}
