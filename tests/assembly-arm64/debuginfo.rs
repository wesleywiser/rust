//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -g -Coverflow-checks=off
//! With `-g`, the backend threads source locations through the instruction stream as `.loc`
//! markers (`Inst::DebugLoc`). The textual asm path renders them as comments; the binary path turns
//! them into a `.debug_line` line-number program and per-function `DW_TAG_subprogram` DIEs, so lldb
//! recovers function names, file/line locations, line breakpoints, and accurate panic backtraces.
//!
//! This guards the line-tracking path: each statement must emit a marker carrying its *exact*
//! source line. Expected lines use FileCheck's `[[@LINE+N]]` so they survive the test being moved.
//! The functions are deliberately call-free so no inlined callee markers appear between them.
#![crate_type = "lib"]

// The three statements below are on consecutive lines; their markers must report those lines in
// order. Each `// CHECK` sits 5 lines above the statement it matches.
// CHECK-LABEL: _dbg_lines:
// CHECK: // .loc file{{[0-9]+}} [[@LINE+5]] {{[0-9]+}}
// CHECK: // .loc file{{[0-9]+}} [[@LINE+5]] {{[0-9]+}}
// CHECK: // .loc file{{[0-9]+}} [[@LINE+5]] {{[0-9]+}}
#[no_mangle]
pub extern "C" fn dbg_lines(a: i64, b: i64) -> i64 {
    let c = a + b;
    let d = c * 3;
    d - 1
}

// A second function confirms locations are tracked per function and the line counter advances to
// this function's own (much later) lines rather than carrying over from the first.
// CHECK-LABEL: _dbg_other:
// CHECK: // .loc file{{[0-9]+}} [[@LINE+4]] {{[0-9]+}}
// CHECK: // .loc file{{[0-9]+}} [[@LINE+4]] {{[0-9]+}}
#[no_mangle]
pub extern "C" fn dbg_other(x: i64) -> i64 {
    let y = x + 100;
    y * 2
}


