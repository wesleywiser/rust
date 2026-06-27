//! Global inline-assembly codegen and symbol mangling (`AsmCodegenMethods`).

use rustc_ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_codegen_ssa::traits::{AsmCodegenMethods, GlobalAsmOperandRef};
use rustc_middle::ty::Instance;
use rustc_span::Span;

use crate::context::CodegenCx;

impl<'tcx> AsmCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn codegen_global_asm(
        &mut self,
        _template: &[InlineAsmTemplatePiece],
        _operands: &[GlobalAsmOperandRef<'tcx>],
        _options: InlineAsmOptions,
        _line_spans: &[Span],
    ) {
        todo!("rustc_codegen_arm64: codegen_global_asm")
    }

    fn mangled_name(&self, instance: Instance<'tcx>) -> String {
        self.mangle(self.tcx.symbol_name(instance).name)
    }
}
