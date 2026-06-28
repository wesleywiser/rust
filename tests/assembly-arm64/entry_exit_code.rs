//@ assembly-output: emit-asm
//@ only-aarch64
//@ only-macos
//@ compile-flags: -Zcodegen-backend=arm64 -Coverflow-checks=off
//! The synthesized C `main` wrapper calls the `lang_start` lang item with only a function type and
//! no `FnAbi`. The backend must still collect `lang_start`'s return value — the process exit code —
//! and return it from the wrapper. The pre-fix backend returned `Undef` for any call made without a
//! `FnAbi`, so the wrapper discarded the result and emitted `movz w0, #0`, making every program
//! exit 0 regardless of what `main` returned (this silently broke ripgrep's exit status, and any
//! `main() -> ExitCode` / `Result`).
#![crate_type = "bin"]

use std::process::ExitCode;

// The wrapper must use the value returned by `lang_start` (kept in `x0`/`w0`), and must not zero
// `w0` between the call and the return.
// CHECK-LABEL: _main:
// CHECK: bl {{.*}}lang_start
// CHECK-NOT: movz w0
// CHECK: ret
fn main() -> ExitCode {
    ExitCode::from(std::hint::black_box(3))
}
