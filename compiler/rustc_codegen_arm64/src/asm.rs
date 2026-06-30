//! Global inline-assembly codegen and symbol mangling (`AsmCodegenMethods`).

use rustc_ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_codegen_ssa::back::symbol_export::escape_symbol_name;
use rustc_codegen_ssa::traits::{AsmCodegenMethods, GlobalAsmOperandRef};
use rustc_middle::ty::Instance;
use rustc_span::Span;

use crate::context::CodegenCx;

impl<'tcx> AsmCodegenMethods<'tcx> for CodegenCx<'tcx> {
    /// Lower a `global_asm!` block (also the path for `#[naked]` function bodies, which
    /// `rustc_codegen_ssa` rewrites into a `global_asm!`-shaped template) by substituting its
    /// operands into the template text and appending it to this codegen unit's accumulated
    /// assembly. The text is handed to an external assembler at emit time
    /// (see [`crate::mach::emit_obj`]'s caller in `lib.rs`).
    fn codegen_global_asm(
        &mut self,
        template: &[InlineAsmTemplatePiece],
        operands: &[GlobalAsmOperandRef<'tcx>],
        _options: InlineAsmOptions,
        _line_spans: &[Span],
    ) {
        let mut asm = self.global_asm.borrow_mut();
        asm.push('\n');
        for piece in template {
            match piece {
                InlineAsmTemplatePiece::String(s) => asm.push_str(s),
                InlineAsmTemplatePiece::Placeholder { operand_idx, modifier: _, span } => {
                    match operands[*operand_idx] {
                        GlobalAsmOperandRef::Const { ref string } => asm.push_str(string),
                        GlobalAsmOperandRef::SymFn { instance } => {
                            let name = self.mangle(self.tcx.symbol_name(instance).name);
                            self.asm_syms.borrow_mut().insert(name.clone().into());
                            asm.push_str(&escape_symbol_name(self.tcx, &name, *span));
                        }
                        GlobalAsmOperandRef::SymStatic { def_id } => {
                            let instance = Instance::mono(self.tcx, def_id);
                            let name = self.mangle(self.tcx.symbol_name(instance).name);
                            self.asm_syms.borrow_mut().insert(name.clone().into());
                            asm.push_str(&escape_symbol_name(self.tcx, &name, *span));
                        }
                    }
                }
            }
        }
        asm.push('\n');
    }

    fn mangled_name(&self, instance: Instance<'tcx>) -> String {
        self.mangle(self.tcx.symbol_name(instance).name)
    }
}

