//! Static item codegen (`StaticCodegenMethods`).

use rustc_codegen_ssa::traits::StaticCodegenMethods;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::interpret::ConstAllocation;

use crate::context::{CodegenCx, TypeData, Value};

impl<'tcx> StaticCodegenMethods for CodegenCx<'tcx> {
    fn static_addr_of(&self, alloc: ConstAllocation<'_>, _kind: Option<&str>) -> Value {
        let sym = self.const_alloc_addr(alloc);
        Value::Sym { sym, offset: 0, ty: self.intern_type(TypeData::Ptr) }
    }

    fn codegen_static(&mut self, def_id: DefId) {
        self.codegen_static_item(def_id);
    }
}
