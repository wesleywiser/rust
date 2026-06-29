//! Per-function machine code: an [`Inst`] sequence plus the layout pass that resolves local
//! branches and records relocations, yielding encoded bytes ready for object/assembly emission.
//!
//! A `MachFunction` is a self-contained, position-independent *atom*: it references everything
//! external (other functions, statics) only by symbol via relocations, never by absolute offset.
//! This is what makes future per-function parallel codegen and binary-patching incremental
//! compilation possible (see the binary-patching compatibility invariants in the design notes).

use std::collections::HashMap;

use crate::mach::inst::{Inst, SymRef, patch_imm19, patch_imm26};

/// The kind of relocation a symbol-referencing instruction needs. Mirrors the AArch64 Mach-O
/// relocation types; the object emitter maps these to `ARM64_RELOC_*`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RelocKind {
    /// `bl`/`b` to a symbol: `ARM64_RELOC_BRANCH26`.
    Branch26,
    /// `adrp` page: `ARM64_RELOC_PAGE21`.
    Page21,
    /// `add`/`ldr` low 12 bits: `ARM64_RELOC_PAGEOFF12`.
    PageOff12,
    /// `adrp` of a GOT entry: `ARM64_RELOC_GOT_LOAD_PAGE21`.
    GotLoadPage21,
    /// `ldr` of a GOT entry's low bits: `ARM64_RELOC_GOT_LOAD_PAGEOFF12`.
    GotLoadPageOff12,
    /// 64-bit absolute pointer in data: `ARM64_RELOC_UNSIGNED`.
    Unsigned64,
    /// `adrp` page of a thread-local variable descriptor: `ARM64_RELOC_TLVP_LOAD_PAGE21`.
    TlvpPage21,
    /// `ldr` of a thread-local descriptor's low bits: `ARM64_RELOC_TLVP_LOAD_PAGEOFF12`.
    TlvpPageOff12,
    /// First half of a `B - A` pair: `ARM64_RELOC_SUBTRACTOR` (subtrahend). Used in `__eh_frame`.
    Subtractor64,
    /// 32-bit subtractor for sdata4 fields in `__eh_frame`.
    Subtractor32,
    /// 32-bit absolute pointer (pairs with `Subtractor32` for pcrel sdata4 fields).
    Unsigned32,
    /// 32-bit pcrel reference to a GOT slot: `ARM64_RELOC_POINTER_TO_GOT`. Used for personality.
    PointerToGot32,
}

/// A relocation to apply at `offset` bytes into a function's (or data item's) contents.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Reloc {
    pub offset: u64,
    pub sym: Box<str>,
    pub addend: i64,
    pub kind: RelocKind,
}

/// A function being built: a flat instruction stream (including [`Inst::Label`] pseudo-ops).
pub struct MachFunction {
    pub name: Box<str>,
    pub is_global: bool,
    pub insts: Vec<Inst>,
    /// Exception-handling call sites: each is `(begin, end, landing_pad, action)` as label ids; a
    /// call whose return address is in `[begin, end)` that unwinds transfers to `landing_pad`.
    /// `action` is 0 for a cleanup pad and 1 for a catch-all (catch_unwind). Empty for non-EH fns.
    pub call_sites: Vec<MachCallSite>,
}

/// An EH call site recorded with label ids (resolved to byte offsets by `encode`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MachCallSite {
    pub begin: u32,
    pub end: u32,
    pub landing_pad: u32,
    pub action: u8,
}

/// A resolved EH call site: byte offsets within the function plus the cleanup action.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CallSite {
    pub start: u32,
    pub len: u32,
    pub lp: u32,
    pub action: u8,
}

/// A function after layout: encoded little-endian code bytes plus the relocations to apply.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EncodedFunction {
    pub name: Box<str>,
    pub is_global: bool,
    pub code: Vec<u8>,
    pub relocs: Vec<Reloc>,
    /// Resolved EH call sites (byte offsets). Non-empty iff the function needs a landing-pad table.
    pub call_sites: Vec<CallSite>,
}

impl MachFunction {
    pub fn new(name: impl Into<Box<str>>, is_global: bool) -> MachFunction {
        MachFunction { name: name.into(), is_global, insts: Vec::new(), call_sites: Vec::new() }
    }

    #[inline]
    pub fn push(&mut self, inst: Inst) {
        self.insts.push(inst);
    }

    /// Resolve labels and relocations, producing encoded bytes.
    ///
    /// Pass 1 assigns each real instruction a byte offset and records label positions. Pass 2
    /// encodes each instruction, patching label-relative branch displacements and recording a
    /// [`Reloc`] for each symbol reference.
    pub fn encode(&self) -> EncodedFunction {
        // Pass 1: label -> byte offset. Labels occupy no space; every other inst is 4 bytes.
        let mut label_offsets: HashMap<u32, u64> = HashMap::new();
        let mut offset: u64 = 0;
        for inst in &self.insts {
            match inst {
                Inst::Label(l) => {
                    label_offsets.insert(*l, offset);
                }
                _ => offset += 4,
            }
        }

        // Pass 2: encode + patch + relocate.
        let mut code = Vec::with_capacity(offset as usize);
        let mut relocs = Vec::new();
        let mut cur: u64 = 0;
        for inst in &self.insts {
            match inst {
                Inst::Label(_) => continue,
                Inst::B { target } => {
                    let word = patch_imm26(
                        inst.encode(),
                        branch_disp(&label_offsets, *target, cur, 26),
                    );
                    code.extend_from_slice(&word.to_le_bytes());
                }
                Inst::BCond { target, .. } | Inst::CbNz { target, .. } => {
                    let word = patch_imm19(
                        inst.encode(),
                        branch_disp(&label_offsets, *target, cur, 19),
                    );
                    code.extend_from_slice(&word.to_le_bytes());
                }
                Inst::Bl { sym } => {
                    push_reloc(&mut relocs, cur, sym, RelocKind::Branch26);
                    code.extend_from_slice(&inst.encode().to_le_bytes());
                }
                Inst::Adrp { sym, .. } => {
                    push_reloc(&mut relocs, cur, sym, RelocKind::Page21);
                    code.extend_from_slice(&inst.encode().to_le_bytes());
                }
                Inst::AddLo { sym, .. } => {
                    push_reloc(&mut relocs, cur, sym, RelocKind::PageOff12);
                    code.extend_from_slice(&inst.encode().to_le_bytes());
                }
                Inst::AdrpTlv { sym, .. } => {
                    push_reloc(&mut relocs, cur, sym, RelocKind::TlvpPage21);
                    code.extend_from_slice(&inst.encode().to_le_bytes());
                }
                Inst::LdrTlvLo { sym, .. } => {
                    push_reloc(&mut relocs, cur, sym, RelocKind::TlvpPageOff12);
                    code.extend_from_slice(&inst.encode().to_le_bytes());
                }
                _ => code.extend_from_slice(&inst.encode().to_le_bytes()),
            }
            cur += 4;
        }

        // Resolve EH call sites: label ids -> byte offsets within the function.
        let call_sites = self
            .call_sites
            .iter()
            .map(|cs| {
                let start = label_offsets[&cs.begin] as u32;
                let end = label_offsets[&cs.end] as u32;
                CallSite { start, len: end - start, lp: label_offsets[&cs.landing_pad] as u32, action: cs.action }
            })
            .collect();

        EncodedFunction { name: self.name.clone(), is_global: self.is_global, code, relocs, call_sites }
    }
}

fn push_reloc(relocs: &mut Vec<Reloc>, offset: u64, sym: &SymRef, kind: RelocKind) {
    relocs.push(Reloc { offset, sym: sym.name.clone(), addend: sym.addend, kind });
}

/// Compute a branch displacement (in instructions) from `cur` to `target`'s label, asserting it
/// fits in a signed field of `bits` width.
fn branch_disp(labels: &HashMap<u32, u64>, target: u32, cur: u64, bits: u32) -> i32 {
    let dest = *labels.get(&target).expect("branch to undefined label");
    let disp = (dest as i64 - cur as i64) / 4;
    let limit = 1i64 << (bits - 1);
    assert!(disp >= -limit && disp < limit, "branch displacement out of range");
    disp as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mach::inst::Inst;
    use crate::mach::reg::{Cond, LR, OperandSize, X0};

    #[test]
    fn forward_and_backward_branches() {
        // .L0: cbz x0, .L1 ; b .L0 ; .L1: ret
        let mut f = MachFunction::new("t", true);
        f.push(Inst::Label(0));
        f.push(Inst::CbNz { nonzero: false, size: OperandSize::S64, rt: X0, target: 1 });
        f.push(Inst::B { target: 0 });
        f.push(Inst::Label(1));
        f.push(Inst::Ret { rn: LR });
        let enc = f.encode();
        assert_eq!(enc.code.len(), 12);

        // cbz x0, .L1: .L1 is at byte 8, cbz at byte 0 -> disp = +2 insns.
        let cbz = u32::from_le_bytes(enc.code[0..4].try_into().unwrap());
        assert_eq!((cbz >> 5) & 0x7FFFF, 2);
        // b .L0: .L0 at byte 0, b at byte 4 -> disp = -1 insn -> imm26 = 0x3FFFFFF.
        let b = u32::from_le_bytes(enc.code[4..8].try_into().unwrap());
        assert_eq!(b & 0x03FF_FFFF, 0x03FF_FFFF);
        assert!(enc.relocs.is_empty());
    }

    #[test]
    fn call_records_branch26_reloc() {
        let mut f = MachFunction::new("caller", true);
        f.push(Inst::Bl { sym: SymRef::new("_callee") });
        f.push(Inst::Ret { rn: LR });
        let enc = f.encode();
        assert_eq!(enc.relocs.len(), 1);
        assert_eq!(enc.relocs[0].offset, 0);
        assert_eq!(&*enc.relocs[0].sym, "_callee");
        assert_eq!(enc.relocs[0].kind, RelocKind::Branch26);
        let _ = Cond::Eq;
    }
}
