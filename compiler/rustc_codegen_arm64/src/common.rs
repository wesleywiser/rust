//! Constant value construction (`ConstCodegenMethods`).

use rustc_abi::{self as abi, Size};
use rustc_codegen_ssa::traits::{
    BaseTypeCodegenMethods, ConstCodegenMethods, MiscCodegenMethods,
};
use rustc_middle::mir::interpret::{GlobalAlloc, Scalar};

use crate::context::{CodegenCx, Type, TypeData, Value};
use crate::mach::frame::align_up;
use crate::mach::func::{Reloc, RelocKind};
use crate::mach::module::{DataItem, DataSection};

impl<'tcx> CodegenCx<'tcx> {
    /// Mask `bits` to the width of an integer type so constants are stored canonically.
    fn mask_to_type(&self, bits: u128, ty: Type) -> u128 {
        match self.type_data(ty) {
            TypeData::Int(width) if width < 128 => bits & ((1u128 << width) - 1),
            _ => bits,
        }
    }

    fn const_int_of(&self, ty: Type, bits: u128) -> Value {
        Value::Const { bits: self.mask_to_type(bits, ty), ty }
    }
}

impl<'tcx> ConstCodegenMethods for CodegenCx<'tcx> {
    fn const_null(&self, t: Type) -> Value {
        Value::Const { bits: 0, ty: t }
    }
    fn const_undef(&self, t: Type) -> Value {
        Value::Undef { ty: t }
    }
    fn const_poison(&self, t: Type) -> Value {
        Value::Undef { ty: t }
    }

    fn const_bool(&self, val: bool) -> Value {
        Value::Const { bits: val as u128, ty: self.intern_type(TypeData::Int(1)) }
    }

    fn const_i8(&self, i: i8) -> Value {
        self.const_int_of(self.type_i8(), i as i64 as u128)
    }
    fn const_i16(&self, i: i16) -> Value {
        self.const_int_of(self.type_i16(), i as i64 as u128)
    }
    fn const_i32(&self, i: i32) -> Value {
        self.const_int_of(self.type_i32(), i as i64 as u128)
    }
    fn const_i64(&self, i: i64) -> Value {
        self.const_int_of(self.type_i64(), i as u128)
    }
    fn const_int(&self, t: Type, i: i64) -> Value {
        self.const_int_of(t, i as u128)
    }
    fn const_u8(&self, i: u8) -> Value {
        self.const_int_of(self.type_i8(), i as u128)
    }
    fn const_u32(&self, i: u32) -> Value {
        self.const_int_of(self.type_i32(), i as u128)
    }
    fn const_u64(&self, i: u64) -> Value {
        self.const_int_of(self.type_i64(), i as u128)
    }
    fn const_u128(&self, i: u128) -> Value {
        self.const_int_of(self.type_i128(), i)
    }
    fn const_usize(&self, i: u64) -> Value {
        self.const_int_of(self.type_isize(), i as u128)
    }
    fn const_uint(&self, t: Type, i: u64) -> Value {
        self.const_int_of(t, i as u128)
    }
    fn const_uint_big(&self, t: Type, u: u128) -> Value {
        self.const_int_of(t, u)
    }

    fn const_real(&self, t: Type, val: f64) -> Value {
        let bits = match self.type_data(t) {
            TypeData::Float(32) => (val as f32).to_bits() as u128,
            _ => val.to_bits() as u128,
        };
        Value::Const { bits, ty: t }
    }

    fn const_str(&self, s: &str) -> (Value, Value) {
        let name = self.mangle(&self.generate_local_symbol_name("str"));
        let sym = self.intern_sym(&name);
        self.module.borrow_mut().push_data(DataItem {
            name: name.into(),
            is_global: false,
            section: DataSection::ReadOnly,
            align: 1,
            bytes: s.as_bytes().to_vec(),
            bss_size: 0,
            relocs: Vec::new(),
        });
        let ptr = Value::Sym { sym, offset: 0, ty: self.intern_type(TypeData::Ptr) };
        let len = self.const_usize(s.len() as u64);
        (ptr, len)
    }

    fn const_struct(&self, elts: &[Value], packed: bool) -> Value {
        // Lay the element constants out into a data item — padding each field to its alignment
        // unless `packed` — and return a pointer to it, typed as an aggregate. Pointer-valued
        // elements (`Value::Sym`) become `UNSIGNED64` relocations with the in-symbol offset left in
        // the data as the Mach-O implicit addend (matching `lower_alloc`). The cg_ssa driver lowers
        // aggregate constants through allocations, so this is only reached by callers that use
        // `const_struct` directly; it is implemented here for completeness so the method is total.
        let mut bytes: Vec<u8> = Vec::new();
        let mut relocs: Vec<Reloc> = Vec::new();
        let mut align: u64 = 1;
        for &elt in elts {
            let (es, ea) = self.type_size_align(elt.ty());
            if !packed {
                bytes.resize(align_up(bytes.len() as u64, ea) as usize, 0);
                align = align.max(ea);
            }
            let field_off = bytes.len() as u64;
            match elt {
                Value::Const { bits, .. } => {
                    bytes.extend_from_slice(&bits.to_le_bytes()[..es as usize]);
                }
                Value::Sym { sym, offset, .. } => {
                    relocs.push(Reloc {
                        offset: field_off,
                        sym: self.sym_name(sym),
                        addend: 0,
                        kind: RelocKind::Unsigned64,
                    });
                    bytes.extend_from_slice(&(offset as u64).to_le_bytes()[..es as usize]);
                }
                // Undef/poison and (invalid here) runtime slot/frame addresses contribute zeroed bytes.
                Value::Undef { .. } | Value::Slot { .. } | Value::FrameAddr { .. } => {
                    bytes.resize(bytes.len() + es as usize, 0);
                }
            }
        }
        let size = if packed { bytes.len() as u64 } else { align_up(bytes.len() as u64, align) };
        bytes.resize(size as usize, 0);
        let agg_ty = self.intern_type(TypeData::Aggregate { size, align });
        let section = if relocs.is_empty() { DataSection::ReadOnly } else { DataSection::Data };
        let name = self.mangle(&self.generate_local_symbol_name("struct"));
        let sym = self.intern_sym(&name);
        self.module.borrow_mut().push_data(DataItem {
            name: name.into(),
            is_global: false,
            section,
            align: align.max(1) as u32,
            bytes,
            bss_size: 0,
            relocs,
        });
        Value::Sym { sym, offset: 0, ty: agg_ty }
    }
    fn const_vector(&self, elts: &[Value]) -> Value {
        // Serialize the lane constants into read-only data and return a pointer to it (typed as the
        // vector). The builder copies this into a frame slot on first use (`vector_to_slot`).
        let mut bytes = Vec::new();
        let mut elem = self.intern_type(TypeData::Int(8));
        for &elt in elts {
            if let Value::Const { bits, ty } = elt {
                elem = ty;
                let (es, _) = self.type_size_align(ty);
                bytes.extend_from_slice(&bits.to_le_bytes()[..es as usize]);
            }
        }
        let vec_ty = self.intern_type(TypeData::Vector(elem, elts.len() as u64));
        let (_, align) = self.type_size_align(vec_ty);
        let name = self.mangle(&self.generate_local_symbol_name("vec"));
        let sym = self.intern_sym(&name);
        self.module.borrow_mut().push_data(DataItem {
            name: name.into(),
            is_global: false,
            section: DataSection::ReadOnly,
            align: align.max(1) as u32,
            bytes,
            bss_size: 0,
            relocs: Vec::new(),
        });
        Value::Sym { sym, offset: 0, ty: vec_ty }
    }

    fn const_to_opt_uint(&self, v: Value) -> Option<u64> {
        match v {
            Value::Const { bits, .. } => u64::try_from(bits).ok(),
            _ => None,
        }
    }
    fn const_to_opt_u128(&self, v: Value, _sign_ext: bool) -> Option<u128> {
        match v {
            Value::Const { bits, .. } => Some(bits),
            _ => None,
        }
    }

    fn scalar_to_backend(&self, cv: Scalar, layout: abi::Scalar, llty: Type) -> Value {
        match cv {
            Scalar::Int(int) => {
                let bits = int.to_bits(layout.size(self));
                Value::Const { bits, ty: llty }
            }
            Scalar::Ptr(ptr, _size) => {
                // A pointer constant: resolve the allocation it points into to a symbol address and
                // carry the in-allocation offset as the symbol addend.
                let (prov, offset) = ptr.prov_and_relative_offset();
                let alloc_id = prov.alloc_id();
                let sym = match self.tcx.global_alloc(alloc_id) {
                    GlobalAlloc::Function { instance, .. } => {
                        let sym = self.function_sym(self.get_fn(instance));
                        if self.tcx.is_foreign_item(instance.def_id()) {
                            self.got_syms.borrow_mut().insert(sym);
                        }
                        sym
                    }
                    GlobalAlloc::Static(def_id) => self.get_static_sym(def_id),
                    GlobalAlloc::Memory(alloc) => self.alloc_symbol(alloc.inner()),
                    GlobalAlloc::VTable(ty, dyn_ty) => {
                        let principal = dyn_ty
                            .principal()
                            .map(|p| self.tcx.instantiate_bound_regions_with_erased(p));
                        let vtable =
                            self.tcx.global_alloc(self.tcx.vtable_allocation((ty, principal)));
                        self.alloc_symbol(vtable.unwrap_memory().inner())
                    }
                    GlobalAlloc::TypeId { .. } => {
                        // A `TypeId` has no real address; model it as an integer constant.
                        return Value::Const { bits: offset.bytes() as u128, ty: llty };
                    }
                };
                Value::Sym { sym, offset: offset.bytes() as i64, ty: llty }
            }
        }
    }

    fn const_ptr_byte_offset(&self, val: Value, offset: Size) -> Value {
        match val {
            Value::Sym { sym, offset: base, ty } => {
                Value::Sym { sym, offset: base + offset.bytes() as i64, ty }
            }
            other => other,
        }
    }
}
