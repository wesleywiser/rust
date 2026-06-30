//! Inline `asm!` lowering (the register-allocated wrapper-call model).
//!
//! This object-emitting backend has no built-in textual assembler, so each `asm!` is turned into a
//! generated *naked wrapper function* (textual AArch64 assembly, assembled later by the platform
//! `cc`; see [`crate::asm`]/`lib.rs`). The wrapper receives a pointer (in `x0`) to a stack
//! "marshalling frame": it loads the inputs from the frame into the allocated registers, runs the
//! user's template, and writes the outputs back to the frame. The call site (in [`crate::builder`])
//! stores inputs into the frame, calls the wrapper, and reads the outputs out.
//!
//! The register allocator, stack-slot layout, and wrapper text are adapted from
//! `rustc_codegen_cranelift`, restricted to AArch64 + Mach-O.

use std::fmt::Write;

use rustc_ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::TyCtxt;
use rustc_span::sym;
use rustc_target::asm::{
    InlineAsmArch, InlineAsmClobberAbi, InlineAsmReg, InlineAsmRegClass, InlineAsmRegOrRegClass,
    allocatable_registers,
};

const ARCH: InlineAsmArch = InlineAsmArch::AArch64;

/// An inline-asm operand, simplified for the wrapper generator (it carries register classes and the
/// textual substitutions, but no SSA values — the call site in `builder.rs` handles those).
pub enum AsmOperand {
    In { reg: InlineAsmRegOrRegClass },
    Out { reg: InlineAsmRegOrRegClass, late: bool, used: bool },
    InOut { reg: InlineAsmRegOrRegClass, late: bool, out_used: bool },
    Const { value: String },
    Sym { value: String },
}

/// The result of laying out an `asm!`: the register allocated to each operand and the marshalling
/// frame offsets for its input/output, plus the total frame size.
pub struct AsmLayout {
    pub regs: Vec<Option<InlineAsmReg>>,
    pub in_slots: Vec<Option<u64>>,
    pub out_slots: Vec<Option<u64>>,
    pub slot_size: u64,
}

/// Lay out and generate the wrapper for one `asm!`. Returns the wrapper's assembly text (to be
/// appended to the codegen unit's global asm) and the marshalling layout used by the call site.
/// `asm_name` is the wrapper's logical symbol name (without the Mach-O leading underscore).
pub fn generate(
    tcx: TyCtxt<'_>,
    enclosing_def_id: DefId,
    asm_name: &str,
    template: &[InlineAsmTemplatePiece],
    operands: &[AsmOperand],
    options: InlineAsmOptions,
) -> (String, AsmLayout) {
    let mut g = Generator {
        tcx,
        enclosing_def_id,
        template,
        operands,
        options,
        regs: vec![None; operands.len()],
        slots_clobber: vec![None; operands.len()],
        slots_input: vec![None; operands.len()],
        slots_output: vec![None; operands.len()],
        slot_size: 0,
    };
    g.allocate_registers();
    g.allocate_stack_slots();
    let text = g.generate_wrapper(asm_name);
    let layout = AsmLayout {
        regs: g.regs,
        in_slots: g.slots_input,
        out_slots: g.slots_output,
        slot_size: g.slot_size,
    };
    (text, layout)
}

struct Generator<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    enclosing_def_id: DefId,
    template: &'a [InlineAsmTemplatePiece],
    operands: &'a [AsmOperand],
    options: InlineAsmOptions,
    regs: Vec<Option<InlineAsmReg>>,
    slots_clobber: Vec<Option<u64>>,
    slots_input: Vec<Option<u64>>,
    slots_output: Vec<Option<u64>>,
    slot_size: u64,
}

fn align_to(offset: u64, align: u64) -> u64 {
    (offset + align - 1) / align * align
}

impl Generator<'_, '_> {
    /// The byte size (= alignment) of the marshalling slot for a register class: the largest type
    /// the class can hold (8 for general-purpose, 16 for SIMD&FP).
    fn slot_bytes(&self, class: InlineAsmRegClass) -> u64 {
        class
            .supported_types(ARCH, true)
            .iter()
            .map(|(ty, _)| ty.size().bytes())
            .max()
            .unwrap_or(8)
    }

    fn allocate_registers(&mut self) {
        let sess = self.tcx.sess;
        let map = allocatable_registers(
            ARCH,
            sess.relocation_model(),
            self.tcx.asm_target_features(self.enclosing_def_id),
            &sess.target,
        );
        // `(used_as_input, used_as_output)` per already-claimed register.
        let mut allocated = FxHashMap::<InlineAsmReg, (bool, bool)>::default();
        let mut regs = vec![None; self.operands.len()];

        // Claim explicitly-named registers first.
        for (i, operand) in self.operands.iter().enumerate() {
            match *operand {
                AsmOperand::In { reg: InlineAsmRegOrRegClass::Reg(reg) } => {
                    regs[i] = Some(reg);
                    allocated.entry(reg).or_default().0 = true;
                }
                AsmOperand::Out { reg: InlineAsmRegOrRegClass::Reg(reg), late: true, .. } => {
                    regs[i] = Some(reg);
                    allocated.entry(reg).or_default().1 = true;
                }
                AsmOperand::Out { reg: InlineAsmRegOrRegClass::Reg(reg), .. }
                | AsmOperand::InOut { reg: InlineAsmRegOrRegClass::Reg(reg), .. } => {
                    regs[i] = Some(reg);
                    allocated.insert(reg, (true, true));
                }
                _ => {}
            }
        }

        // Allocate out/inout (more constrained) before in/lateout.
        for (i, operand) in self.operands.iter().enumerate() {
            match *operand {
                AsmOperand::Out { reg: InlineAsmRegOrRegClass::RegClass(class), late: false, .. }
                | AsmOperand::InOut { reg: InlineAsmRegOrRegClass::RegClass(class), .. } => {
                    let reg = Self::alloc_reg(&map, &allocated, class, false);
                    regs[i] = Some(reg);
                    allocated.insert(reg, (true, true));
                }
                _ => {}
            }
        }

        // Allocate in/lateout.
        for (i, operand) in self.operands.iter().enumerate() {
            match *operand {
                AsmOperand::In { reg: InlineAsmRegOrRegClass::RegClass(class) } => {
                    let reg = Self::alloc_reg(&map, &allocated, class, true);
                    regs[i] = Some(reg);
                    allocated.entry(reg).or_default().0 = true;
                }
                AsmOperand::Out { reg: InlineAsmRegOrRegClass::RegClass(class), late: true, .. } => {
                    let reg = Self::alloc_reg(&map, &allocated, class, false);
                    regs[i] = Some(reg);
                    allocated.entry(reg).or_default().1 = true;
                }
                _ => {}
            }
        }

        self.regs = regs;
    }

    /// Pick the first register of `class` not overlapping an already-claimed register. When
    /// `for_input`, an output-only claim does not conflict.
    fn alloc_reg(
        map: &FxHashMap<InlineAsmRegClass, rustc_data_structures::fx::FxIndexSet<InlineAsmReg>>,
        allocated: &FxHashMap<InlineAsmReg, (bool, bool)>,
        class: InlineAsmRegClass,
        for_input: bool,
    ) -> InlineAsmReg {
        for &reg in &map[&class] {
            let mut used = false;
            reg.overlapping_regs(|r| {
                let conflict = match allocated.get(&r) {
                    Some(&(inp, out)) => {
                        if for_input { inp } else { inp || out }
                    }
                    None => false,
                };
                if conflict {
                    used = true;
                }
            });
            if !used {
                return reg;
            }
        }
        panic!("rustc_codegen_arm64: cannot allocate inline-asm register for {class:?}")
    }

    fn allocate_stack_slots(&mut self) {
        let mut slot_size = 0u64;

        // Save slots for registers clobbered by the asm that are NOT already clobbered by the C ABI
        // (those are caller-saved across our call anyway).
        let abi_clobber = InlineAsmClobberAbi::parse(
            ARCH,
            &self.tcx.sess.target,
            &self.tcx.sess.unstable_target_features,
            sym::C,
        )
        .unwrap()
        .clobbered_regs();
        for i in 0..self.operands.len() {
            let Some(reg) = self.regs[i] else { continue };
            let mut abi_clobbered = false;
            for &c in abi_clobber {
                c.overlapping_regs(|r| {
                    if r == reg {
                        abi_clobbered = true;
                    }
                });
                if abi_clobbered {
                    break;
                }
            }
            if !abi_clobbered {
                let bytes = self.slot_bytes(reg.reg_class());
                let off = align_to(slot_size, bytes);
                slot_size = off + bytes;
                self.slots_clobber[i] = Some(off);
            }
        }

        // inout operands with an output place share a single slot for input and output.
        for (i, operand) in self.operands.iter().enumerate() {
            if let AsmOperand::InOut { reg, out_used: true, .. } = *operand {
                let bytes = self.slot_bytes(reg.reg_class());
                let off = align_to(slot_size, bytes);
                slot_size = off + bytes;
                self.slots_input[i] = Some(off);
                self.slots_output[i] = Some(off);
            }
        }

        let slot_size_before_input = slot_size;

        // Input slots.
        for (i, operand) in self.operands.iter().enumerate() {
            match *operand {
                AsmOperand::In { reg } | AsmOperand::InOut { reg, out_used: false, .. } => {
                    let bytes = self.slot_bytes(reg.reg_class());
                    let off = align_to(slot_size, bytes);
                    slot_size = off + bytes;
                    self.slots_input[i] = Some(off);
                }
                _ => {}
            }
        }

        let slot_size_after_input = slot_size;
        // Output-only slots may reuse the input region (inputs are consumed before outputs).
        slot_size = slot_size_before_input;
        for (i, operand) in self.operands.iter().enumerate() {
            if let AsmOperand::Out { reg, used: true, .. } = *operand {
                let bytes = self.slot_bytes(reg.reg_class());
                let off = align_to(slot_size, bytes);
                slot_size = off + bytes;
                self.slots_output[i] = Some(off);
            }
        }
        slot_size = slot_size.max(slot_size_after_input);

        self.slot_size = slot_size;
    }

    fn generate_wrapper(&self, asm_name: &str) -> String {
        let mut out = String::new();
        // Mach-O: the symbol gets a leading underscore.
        let _ = writeln!(out, ".globl _{asm_name}");
        let _ = writeln!(out, "_{asm_name}:");

        prologue(&mut out);

        if !self.options.contains(InlineAsmOptions::NORETURN) {
            for (reg, slot) in self.iter_slots(&self.slots_clobber) {
                save_register(&mut out, reg, slot);
            }
        }
        for (reg, slot) in self.iter_slots(&self.slots_input) {
            restore_register(&mut out, reg, slot);
        }

        // Enable any non-baseline target features so their instructions assemble.
        for feature in &self.tcx.codegen_fn_attrs(self.enclosing_def_id).target_features {
            if feature.name != sym::neon {
                let _ = writeln!(out, ".arch_extension {}", feature.name);
            }
        }

        // The user's template, with placeholders substituted.
        for piece in self.template {
            match piece {
                InlineAsmTemplatePiece::String(s) => out.push_str(s),
                InlineAsmTemplatePiece::Placeholder { operand_idx, modifier, span: _ } => {
                    match &self.operands[*operand_idx] {
                        AsmOperand::In { .. }
                        | AsmOperand::Out { .. }
                        | AsmOperand::InOut { .. } => {
                            let reg = self.regs[*operand_idx].unwrap();
                            let _ = reg.emit(&mut out, ARCH, *modifier);
                        }
                        AsmOperand::Const { value } => out.push_str(value),
                        AsmOperand::Sym { value } => out.push_str(value),
                    }
                }
            }
        }
        out.push('\n');

        for feature in &self.tcx.codegen_fn_attrs(self.enclosing_def_id).target_features {
            if feature.name != sym::neon {
                let _ = writeln!(out, ".arch_extension no{}", feature.name);
            }
        }

        if !self.options.contains(InlineAsmOptions::NORETURN) {
            for (reg, slot) in self.iter_slots(&self.slots_output) {
                save_register(&mut out, reg, slot);
            }
            for (reg, slot) in self.iter_slots(&self.slots_clobber) {
                restore_register(&mut out, reg, slot);
            }
            epilogue(&mut out);
        } else {
            // `brk #1` — a noreturn asm must not fall through.
            out.push_str("    brk #0x1\n");
        }
        out.push_str("\n\n");
        out
    }

    /// Iterate `(reg, slot_offset)` for the operands that have both an allocated register and a slot
    /// in `slots`.
    fn iter_slots<'s>(
        &'s self,
        slots: &'s [Option<u64>],
    ) -> impl Iterator<Item = (InlineAsmReg, u64)> + 's {
        self.regs
            .iter()
            .zip(slots.iter())
            .filter_map(|(r, s)| r.zip(*s))
    }
}

fn prologue(out: &mut String) {
    // Save fp/lr and the base pointer x19; receive the marshalling-frame pointer (x0) into x19.
    // x19 is reserved by rustc (LLVM's base pointer), so the asm template can't use it.
    out.push_str("    stp fp, lr, [sp, #-32]!\n");
    out.push_str("    mov fp, sp\n");
    out.push_str("    str x19, [sp, #24]\n");
    out.push_str("    mov x19, x0\n");
}

fn epilogue(out: &mut String) {
    out.push_str("    ldr x19, [sp, #24]\n");
    out.push_str("    ldp fp, lr, [sp], #32\n");
    out.push_str("    ret\n");
}

fn save_register(out: &mut String, reg: InlineAsmReg, offset: u64) {
    out.push_str("    str ");
    emit_reg_full(out, reg);
    let _ = writeln!(out, ", [x19, #{offset}]");
}

fn restore_register(out: &mut String, reg: InlineAsmReg, offset: u64) {
    out.push_str("    ldr ");
    emit_reg_full(out, reg);
    let _ = writeln!(out, ", [x19, #{offset}]");
}

/// Emit a register name at its full marshalling width: `q<n>` for a SIMD&FP register (rustc names
/// them `v<n>`), the plain 64-bit name otherwise.
fn emit_reg_full(out: &mut String, reg: InlineAsmReg) {
    match reg {
        InlineAsmReg::AArch64(r) if r.vreg_index().is_some() => {
            let _ = r.emit(out, ARCH, Some('q'));
        }
        _ => {
            let _ = reg.emit(out, ARCH, None);
        }
    }
}
