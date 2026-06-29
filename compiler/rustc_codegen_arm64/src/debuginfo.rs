//! Debug-info trait methods (`DebugInfoCodegenMethods`).
//!
//! Phase 1 emits per-function `DW_TAG_subprogram` DIEs and a line-number program (see
//! [`crate::dwarf`]). Type and variable DIEs are not produced yet, so `create_dbg_var` /
//! `dbg_var_addr` remain no-ops and `DIVariable` is `()`. These methods are only ever called by the
//! SSA driver when debug info is enabled, so the `DebugContext` is always present here.

use gimli::write::UnitEntryId;
use rustc_codegen_ssa::mir::debuginfo::VariableKind;
use rustc_codegen_ssa::traits::DebugInfoCodegenMethods;
use rustc_middle::ty::{ExistentialTraitRef, Instance, Ty};
use rustc_span::{BytePos, SourceFile, Span, Symbol};
use rustc_target::callconv::FnAbi;

use crate::context::{CodegenCx, Function, Value};
use crate::mach::inst::DebugLoc;

impl<'tcx> DebugInfoCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn create_vtable_debuginfo(
        &self,
        _ty: Ty<'tcx>,
        _trait_ref: Option<ExistentialTraitRef<'tcx>>,
        _vtable: Value,
    ) {
    }

    fn dbg_create_lexical_block(&self, _pos: BytePos, parent_scope: UnitEntryId) -> UnitEntryId {
        // Phase 1 does not emit lexical-block DIEs; collapse blocks into their parent scope.
        parent_scope
    }

    fn dbg_location_clone_with_discriminator(
        &self,
        loc: DebugLoc,
        _discriminator: u32,
    ) -> Option<DebugLoc> {
        // Discriminators are not modeled yet; reuse the same location.
        Some(loc)
    }

    fn dbg_scope_fn(
        &self,
        instance: Instance<'tcx>,
        _fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        maybe_definition_llfn: Option<Function>,
    ) -> UnitEntryId {
        let mut debug = self.debug.as_ref().expect("debug info enabled").borrow_mut();
        match maybe_definition_llfn {
            // A function defined in this codegen unit: create its subprogram DIE, keyed by the same
            // mangled symbol name its `MachFunction` carries so the address range can be attached.
            Some(_) => {
                let name = self.mangle(self.tcx.symbol_name(instance).name);
                debug.define_function(self.tcx, instance, &name)
            }
            // A declaration referenced from an inlined frame (Phase 3); use the CU root for now.
            None => debug.root(),
        }
    }

    fn dbg_loc(
        &self,
        _scope: UnitEntryId,
        _inlined_at: Option<DebugLoc>,
        span: Span,
    ) -> DebugLoc {
        self.debug.as_ref().expect("debug info enabled").borrow_mut().source_loc(self.tcx, span)
    }

    fn extend_scope_to_file(&self, scope_metadata: UnitEntryId, _file: &SourceFile) -> UnitEntryId {
        scope_metadata
    }

    fn debuginfo_finalize(&self) {
        // DWARF is serialized during object emission (see `crate::mach::emit_obj`), so there is
        // nothing to finalize here. (This backend's custom CGU driver never calls this anyway.)
    }

    fn create_dbg_var(
        &self,
        _variable_name: Symbol,
        _variable_type: Ty<'tcx>,
        _scope_metadata: UnitEntryId,
        _variable_kind: VariableKind,
        _span: Span,
    ) {
        // Phase 1 does not emit variable DIEs.
    }
}

