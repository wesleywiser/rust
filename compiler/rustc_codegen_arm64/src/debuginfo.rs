//! Debug-info methods. Debug info is not yet emitted, so all of these are no-ops over the unit
//! `DIScope`/`DILocation`/`DIVariable` types declared in `BackendTypes`.

use rustc_codegen_ssa::mir::debuginfo::VariableKind;
use rustc_codegen_ssa::traits::DebugInfoCodegenMethods;
use rustc_middle::ty::{ExistentialTraitRef, Instance, Ty};
use rustc_span::{BytePos, SourceFile, Span, Symbol};
use rustc_target::callconv::FnAbi;

use crate::context::{CodegenCx, Function, Value};

impl<'tcx> DebugInfoCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn create_vtable_debuginfo(
        &self,
        _ty: Ty<'tcx>,
        _trait_ref: Option<ExistentialTraitRef<'tcx>>,
        _vtable: Value,
    ) {
    }

    fn dbg_create_lexical_block(&self, _pos: BytePos, _parent_scope: ()) {}

    fn dbg_location_clone_with_discriminator(
        &self,
        _loc: (),
        _discriminator: u32,
    ) -> Option<()> {
        None
    }

    fn dbg_scope_fn(
        &self,
        _instance: Instance<'tcx>,
        _fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        _maybe_definition_llfn: Option<Function>,
    ) {
    }

    fn dbg_loc(&self, _scope: (), _inlined_at: Option<()>, _span: Span) {}

    fn extend_scope_to_file(&self, _scope_metadata: (), _file: &SourceFile) {}

    fn debuginfo_finalize(&self) {}

    fn create_dbg_var(
        &self,
        _variable_name: Symbol,
        _variable_type: Ty<'tcx>,
        _scope_metadata: (),
        _variable_kind: VariableKind,
        _span: Span,
    ) {
    }
}
