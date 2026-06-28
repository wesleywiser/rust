//! The machine-code (MC) layer: a small AArch64 instruction model with two emitters — a binary
//! Mach-O object encoder and a textual `.s` printer.
//!
//! This layer is deliberately free of `rustc_*` dependencies so it can be reasoned about and
//! unit-tested in isolation, and so the object-format/relocation specifics stay isolated behind it
//! (see the binary-patching compatibility invariants in the design notes).

pub mod emit_asm;
pub mod emit_obj;
pub mod frame;
pub mod func;
pub mod inst;
pub mod module;
pub mod reg;
