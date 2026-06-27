//! Two-phase predefinition of functions and statics (`PreDefineCodegenMethods`).

use rustc_codegen_ssa::traits::PreDefineCodegenMethods;
use rustc_hir::attrs::Linkage;
use rustc_hir::def_id::DefId;
use rustc_middle::mono::Visibility;
use rustc_middle::ty::Instance;

use crate::context::CodegenCx;

/// Whether a symbol with the given linkage is externally visible (placed in the symbol table with
/// global scope) rather than codegen-unit-local.
fn is_global_linkage(linkage: Linkage) -> bool {
    !matches!(linkage, Linkage::Internal)
}

impl<'tcx> PreDefineCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn predefine_static(
        &mut self,
        def_id: DefId,
        _linkage: Linkage,
        _visibility: Visibility,
        symbol_name: &str,
    ) {
        let name = self.mangle(symbol_name);
        let sym = self.intern_sym(&name);
        self.statics.borrow_mut().insert(def_id, sym);
    }

    fn predefine_fn(
        &mut self,
        instance: Instance<'tcx>,
        linkage: Linkage,
        _visibility: Visibility,
        symbol_name: &str,
    ) {
        let name = self.mangle(symbol_name);
        let func = self.declare_named_fn(&name, is_global_linkage(linkage));
        self.instances.borrow_mut().insert(instance, func);
    }
}
