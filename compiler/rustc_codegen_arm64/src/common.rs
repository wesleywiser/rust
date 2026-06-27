//! Constant value construction (`ConstCodegenMethods`).

use rustc_abi::{self as abi, Size};
use rustc_codegen_ssa::traits::{BaseTypeCodegenMethods, ConstCodegenMethods};
use rustc_middle::mir::interpret::Scalar;

use crate::context::{CodegenCx, Type, TypeData, Value};
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

    fn const_struct(&self, _elts: &[Value], _packed: bool) -> Value {
        // Aggregate constants are materialized into data items during static/const-allocation
        // lowering; the scalar `Value` model can't represent them directly.
        todo!("rustc_codegen_arm64: const_struct")
    }
    fn const_vector(&self, _elts: &[Value]) -> Value {
        todo!("rustc_codegen_arm64: const_vector")
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
            Scalar::Ptr(..) => {
                // Pointers into constant allocations require allocation lowering; deferred.
                todo!("rustc_codegen_arm64: scalar_to_backend for pointer scalars")
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
