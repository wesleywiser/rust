//! Static item codegen (`StaticCodegenMethods`).
//!
//! Full const-allocation lowering (turning a `ConstAllocation` into data bytes plus pointer
//! relocations) is deferred; these are stubs that will be filled when statics land.

use rustc_codegen_ssa::traits::StaticCodegenMethods;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::interpret::ConstAllocation;

use crate::context::{CodegenCx, Value};

impl<'tcx> StaticCodegenMethods for CodegenCx<'tcx> {
    fn static_addr_of(&self, _alloc: ConstAllocation<'_>, _kind: Option<&str>) -> Value {
        todo!("rustc_codegen_arm64: static_addr_of")
    }

    fn codegen_static(&mut self, _def_id: DefId) {
        todo!("rustc_codegen_arm64: codegen_static")
    }
}
