//! Lowering constant allocations (`ConstAllocation`) into data items plus pointer relocations.
//!
//! A constant's bytes are emitted verbatim; each pointer in its provenance becomes an
//! `ARM64_RELOC_UNSIGNED` relocation to the referenced symbol (a function, another static, an
//! anonymous constant, or a vtable). The pointer's relative offset stays in the data — Mach-O uses
//! an implicit addend for `UNSIGNED` relocations — so the relocation addend is always zero.

use rustc_hir::def_id::DefId;
use rustc_middle::mir::interpret::{AllocId, Allocation, ConstAllocation, GlobalAlloc};
use rustc_middle::ty::Instance;

use crate::context::{CodegenCx, Sym};
use crate::mach::func::{Reloc, RelocKind};
use crate::mach::module::{DataItem, DataSection};

impl<'tcx> CodegenCx<'tcx> {
    /// The final (mangled) symbol name of a `static` item.
    pub fn static_symbol_name(&self, def_id: DefId) -> Box<str> {
        if let Some(&sym) = self.statics.borrow().get(&def_id) {
            return self.sym_name(sym);
        }
        let instance = Instance::mono(self.tcx, def_id);
        self.mangle(self.tcx.symbol_name(instance).name).into()
    }

    /// The interned symbol for a `static`, creating it on demand (e.g. an extern static defined in
    /// another crate that was never predefined in this codegen unit).
    pub fn get_static_sym(&self, def_id: DefId) -> Sym {
        if let Some(&sym) = self.statics.borrow().get(&def_id) {
            return sym;
        }
        let instance = Instance::mono(self.tcx, def_id);
        let name = self.mangle(self.tcx.symbol_name(instance).name);
        let sym = self.intern_sym(&name);
        self.statics.borrow_mut().insert(def_id, sym);
        sym
    }

    /// Return the symbol of an anonymous constant allocation, emitting it (and, recursively, the
    /// allocations it points to) as data items on first use. Deduplicated by allocation content.
    pub fn alloc_symbol(&self, alloc: &Allocation) -> Sym {
        if let Some(&sym) = self.static_consts.borrow().get(alloc) {
            return sym;
        }
        // Reserve the symbol BEFORE lowering so that a self-referential constant terminates.
        let name = self.mangle(&self.generate_local_symbol_name("anon"));
        let sym = self.intern_sym(&name);
        self.static_consts.borrow_mut().insert(alloc.clone(), sym);

        let (bytes, relocs) = self.lower_alloc(alloc);
        let align = alloc.align.bytes() as u32;
        let mutable = alloc.mutability.is_mut();
        self.push_alloc_data(name.into(), false, mutable, align, bytes, relocs);
        sym
    }

    /// Emit a data item, picking a section: pure read-only bytes go to `__TEXT,__const`, while
    /// anything mutable or carrying relocations goes to `__DATA,__data` (relocated data may not live
    /// in the read-only `__TEXT` segment).
    pub fn push_alloc_data(
        &self,
        name: Box<str>,
        is_global: bool,
        mutable: bool,
        align: u32,
        bytes: Vec<u8>,
        relocs: Vec<Reloc>,
    ) {
        let section = if mutable || !relocs.is_empty() {
            DataSection::Data
        } else {
            DataSection::ReadOnly
        };
        self.module.borrow_mut().push_data(DataItem {
            name,
            is_global,
            section,
            align,
            bytes,
            bss_size: 0,
            relocs,
        });
    }

    /// Lower an allocation into raw initializer bytes plus an `UNSIGNED64` relocation for each
    /// embedded pointer. The pointer's relative offset is left in the data (Mach-O implicit addend).
    pub fn lower_alloc(&self, alloc: &Allocation) -> (Vec<u8>, Vec<Reloc>) {
        let bytes =
            alloc.inspect_with_uninit_and_ptr_outside_interpreter(0..alloc.len()).to_vec();
        let mut relocs = Vec::new();
        for &(offset, prov) in alloc.provenance().ptrs().iter() {
            let Some(sym) = self.global_alloc_target_name(prov.alloc_id()) else { continue };
            relocs.push(Reloc {
                offset: offset.bytes(),
                sym,
                addend: 0,
                kind: RelocKind::Unsigned64,
            });
        }
        (bytes, relocs)
    }

    /// The symbol name that a provenance pointer references — a function, a `static`, an anonymous
    /// constant, or a vtable — or `None` for a `TypeId` provenance, which carries no real address.
    fn global_alloc_target_name(&self, alloc_id: AllocId) -> Option<Box<str>> {
        Some(match self.tcx.global_alloc(alloc_id) {
            GlobalAlloc::Function { instance, .. } => {
                self.mangle(self.tcx.symbol_name(instance).name).into()
            }
            GlobalAlloc::Static(def_id) => self.static_symbol_name(def_id),
            GlobalAlloc::Memory(alloc) => self.sym_name(self.alloc_symbol(alloc.inner())),
            GlobalAlloc::VTable(ty, dyn_ty) => {
                let principal = dyn_ty
                    .principal()
                    .map(|p| self.tcx.instantiate_bound_regions_with_erased(p));
                let vtable = self.tcx.global_alloc(self.tcx.vtable_allocation((ty, principal)));
                self.sym_name(self.alloc_symbol(vtable.unwrap_memory().inner()))
            }
            GlobalAlloc::TypeId { .. } => return None,
        })
    }

    /// Emit a `static` item: lower its initializer allocation and place it under the static's
    /// (global) symbol.
    pub fn codegen_static_item(&self, def_id: DefId) {
        let alloc = match self.tcx.eval_static_initializer(def_id) {
            Ok(alloc) => alloc,
            // A const-eval error here has already been reported during analysis.
            Err(_) => return,
        };
        let inner = alloc.inner();
        let (bytes, relocs) = self.lower_alloc(inner);
        let name = self.static_symbol_name(def_id);
        let align = inner.align.bytes() as u32;
        // Thread-local statics go in `__thread_data`; the object/asm emitters synthesize the
        // `__thread_vars` descriptor that the TLV access sequence (`get_static`) targets.
        if self.tcx.is_thread_local_static(def_id) {
            self.module.borrow_mut().push_data(DataItem {
                name,
                is_global: true,
                section: DataSection::Tls,
                align,
                bytes,
                bss_size: 0,
                relocs,
            });
            return;
        }
        let mutable = inner.mutability.is_mut();
        self.push_alloc_data(name, true, mutable, align, bytes, relocs);
    }
}

impl<'tcx> CodegenCx<'tcx> {
    /// Address of an interned constant allocation (used by `static_addr_of`).
    pub fn const_alloc_addr(&self, alloc: ConstAllocation<'_>) -> Sym {
        self.alloc_symbol(alloc.inner())
    }
}
