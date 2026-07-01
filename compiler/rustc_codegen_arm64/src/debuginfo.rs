//! Debug-info trait methods (`DebugInfoCodegenMethods`).
//!
//! `DW_TAG_subprogram` DIEs and a line-number program are emitted for every function (see
//! [`crate::dwarf`]), together with local-variable and parameter DIEs (`DW_TAG_variable` /
//! `DW_TAG_formal_parameter`) that carry a type and a `DW_OP_fbreg` frame-relative location. Type
//! DIEs cover the scalar base types and thin pointers/references; other types get a correctly-sized
//! opaque placeholder for now. These methods are only ever called by the SSA driver when debug info
//! is enabled, so the `DebugContext` is always present here.

use gimli::write::UnitEntryId;
use rustc_abi::BackendRepr;
use rustc_codegen_ssa::debuginfo::type_names::compute_debuginfo_type_name;
use rustc_codegen_ssa::mir::debuginfo::VariableKind;
use rustc_codegen_ssa::traits::DebugInfoCodegenMethods;
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{self, ExistentialTraitRef, Instance, Ty};
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
        // Give the cloned location a distinct identity so two inlinings sharing a call-site span
        // (e.g. from a macro) form separate inline frames rather than being merged.
        Some(self.debug.as_ref().expect("debug info enabled").borrow_mut().clone_location(loc))
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
            // A callee inlined into the function being codegen'd: create (or reuse) its abstract
            // subprogram DIE, which `DW_TAG_inlined_subroutine`s reference via `abstract_origin`.
            None => debug.define_abstract_function(self.tcx, instance),
        }
    }

    fn dbg_loc(
        &self,
        scope: UnitEntryId,
        inlined_at: Option<DebugLoc>,
        span: Span,
    ) -> DebugLoc {
        self.debug
            .as_ref()
            .expect("debug info enabled")
            .borrow_mut()
            .make_location(self.tcx, scope, inlined_at, span)
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
        variable_name: Symbol,
        variable_type: Ty<'tcx>,
        scope_metadata: UnitEntryId,
        variable_kind: VariableKind,
        span: Span,
    ) -> Option<UnitEntryId> {
        // Only describe variables of a concrete function defined in this codegen unit. A variable
        // whose scope is an *abstract* (inlined-callee) subprogram would attach a concrete,
        // outer-frame `DW_OP_fbreg` location to an abstract template — incorrect in general, and
        // duplicated when the callee is inlined more than once. Proper inline-variable DWARF
        // (concrete instances nested under `DW_TAG_inlined_subroutine`, referencing abstract origins)
        // is left to a later phase; until then such variables are omitted rather than misdescribed.
        if !self.debug_context().borrow().is_concrete_fn(scope_metadata) {
            return None;
        }
        let type_die = self.dbg_type(variable_type);
        let is_parameter = matches!(variable_kind, VariableKind::ArgumentVariable(_));
        Some(self.debug_context().borrow_mut().create_variable(
            self.tcx,
            scope_metadata,
            is_parameter,
            variable_name.as_str(),
            type_die,
            span,
        ))
    }
}

impl<'tcx> CodegenCx<'tcx> {
    /// Get (creating and caching) the DWARF type DIE for `ty`.
    ///
    /// Real DIEs are emitted for the scalar base types and thin pointers/references; every other
    /// type (aggregates, wide pointers, closures, ...) gets a correctly-sized opaque placeholder
    /// whose fields later phases will fill in. The cache lives in the `TyCtxt`-bearing half (keyed
    /// by `Ty`), while the DIEs it points at live in the `DebugContext` that outlives it.
    fn dbg_type(&self, ty: Ty<'tcx>) -> UnitEntryId {
        if let Some(&id) = self.dbg_type_dies.borrow().get(&ty) {
            return id;
        }

        let id = match ty.kind() {
            ty::Bool => self.dbg_base_type(ty, "bool", gimli::DW_ATE_boolean),
            ty::Char => self.dbg_base_type(ty, "char", gimli::DW_ATE_UTF),
            ty::Int(int_ty) => self.dbg_base_type(ty, int_ty.name_str(), gimli::DW_ATE_signed),
            ty::Uint(uint_ty) => self.dbg_base_type(ty, uint_ty.name_str(), gimli::DW_ATE_unsigned),
            ty::Float(float_ty) => self.dbg_base_type(ty, float_ty.name_str(), gimli::DW_ATE_float),

            // Thin pointers/references become `DW_TAG_pointer_type`. A wide (fat) pointer is a
            // scalar pair; it falls through to the placeholder until wide-pointer debuginfo lands.
            ty::Ref(_, pointee, _) | ty::RawPtr(pointee, _)
                if matches!(self.layout_of(ty).backend_repr, BackendRepr::Scalar(_)) =>
            {
                let pointee_die = self.dbg_type(*pointee);
                // A recursive type may have created this pointer's DIE during the recursion above.
                if let Some(&id) = self.dbg_type_dies.borrow().get(&ty) {
                    return id;
                }
                let name = compute_debuginfo_type_name(self.tcx, ty, true);
                let size = self.layout_of(ty).size.bytes();
                self.debug_context().borrow_mut().add_pointer_type(&name, pointee_die, size)
            }

            _ => {
                let name = compute_debuginfo_type_name(self.tcx, ty, true);
                let size = self.layout_of(ty).size.bytes();
                self.debug_context().borrow_mut().add_opaque_type(&name, size)
            }
        };

        self.dbg_type_dies.borrow_mut().insert(ty, id);
        id
    }

    /// Create a `DW_TAG_base_type` DIE for a scalar `ty` (the caller inserts it into the cache).
    fn dbg_base_type(&self, ty: Ty<'tcx>, name: &str, encoding: gimli::DwAte) -> UnitEntryId {
        let size = self.layout_of(ty).size.bytes();
        self.debug_context().borrow_mut().add_base_type(name, encoding, size)
    }

    /// The debug context, present whenever the debug-info methods are invoked.
    pub(crate) fn debug_context(&self) -> &std::cell::RefCell<crate::dwarf::DebugContext> {
        self.debug.as_ref().expect("debug info enabled")
    }
}

