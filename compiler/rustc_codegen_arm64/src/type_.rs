//! Backend type construction and the mapping from `rustc` layouts to backend [`Type`]s.

use rustc_abi::{AddressSpace, BackendRepr, HasDataLayout, Integer, Primitive, Reg, RegKind, Scalar};
use rustc_codegen_ssa::common::TypeKind;
use rustc_codegen_ssa::traits::{
    BaseTypeCodegenMethods, LayoutTypeCodegenMethods, TypeMembershipCodegenMethods,
};
use rustc_middle::ty::Ty;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_target::callconv::{CastTarget, FnAbi};

use crate::context::{CodegenCx, Type, TypeData};

impl<'tcx> CodegenCx<'tcx> {
    /// The backend type for a scalar. In a register, `bool` (and other 0/1 scalars) is `i1`; in
    /// memory it is `i8`.
    fn scalar_backend_type(&self, scalar: Scalar, immediate: bool) -> Type {
        match scalar.primitive() {
            Primitive::Int(int, _signed) => {
                if immediate && scalar.is_bool() {
                    self.intern_type(TypeData::Int(1))
                } else {
                    self.intern_type(TypeData::Int(integer_bits(int)))
                }
            }
            Primitive::Float(float) => {
                self.intern_type(TypeData::Float(float.size().bits() as u32))
            }
            Primitive::Pointer(_) => self.intern_type(TypeData::Ptr),
        }
    }

    /// Map a `rustc` layout to a backend type. `immediate` selects the register representation.
    fn layout_backend_type(&self, layout: TyAndLayout<'tcx>, immediate: bool) -> Type {
        match layout.backend_repr {
            BackendRepr::Scalar(scalar) => self.scalar_backend_type(scalar, immediate),
            BackendRepr::SimdVector { element, count } => {
                let elem = self.scalar_backend_type(element, false);
                self.intern_type(TypeData::Vector(elem, count))
            }
            // A scalar pair packs its two fields into one backend type so the builder can split it
            // across the `PassMode::Pair` registers (and load/store both halves from memory).
            BackendRepr::ScalarPair(a, b) => {
                let a_ty = self.scalar_backend_type(a, immediate);
                let b_ty = self.scalar_backend_type(b, immediate);
                self.intern_type(TypeData::Pair(a_ty, b_ty))
            }
            // Memory aggregates and scalable vectors are addressed by byte offset, so only their
            // size/align matter to the baseline lowering.
            _ => self.intern_type(TypeData::Aggregate {
                size: layout.size.bytes(),
                align: layout.align.abi.bytes(),
            }),
        }
    }
}

fn integer_bits(int: Integer) -> u32 {
    use Integer::*;
    match int {
        I8 => 8,
        I16 => 16,
        I32 => 32,
        I64 => 64,
        I128 => 128,
    }
}

impl<'tcx> BaseTypeCodegenMethods for CodegenCx<'tcx> {
    fn type_i8(&self) -> Type {
        self.intern_type(TypeData::Int(8))
    }
    fn type_i16(&self) -> Type {
        self.intern_type(TypeData::Int(16))
    }
    fn type_i32(&self) -> Type {
        self.intern_type(TypeData::Int(32))
    }
    fn type_i64(&self) -> Type {
        self.intern_type(TypeData::Int(64))
    }
    fn type_i128(&self) -> Type {
        self.intern_type(TypeData::Int(128))
    }
    fn type_isize(&self) -> Type {
        let bits = self.data_layout().pointer_size().bits() as u32;
        self.intern_type(TypeData::Int(bits))
    }

    fn type_f16(&self) -> Type {
        self.intern_type(TypeData::Float(16))
    }
    fn type_f32(&self) -> Type {
        self.intern_type(TypeData::Float(32))
    }
    fn type_f64(&self) -> Type {
        self.intern_type(TypeData::Float(64))
    }
    fn type_f128(&self) -> Type {
        self.intern_type(TypeData::Float(128))
    }

    fn type_array(&self, ty: Type, len: u64) -> Type {
        self.intern_type(TypeData::Array(ty, len))
    }

    fn type_func(&self, _args: &[Type], ret: Type) -> Type {
        // Only the return type is recorded (see `TypeData::Func`); argument types are unused by the
        // baseline, so we avoid the `to_vec` allocation here.
        self.intern_type(TypeData::Func { ret })
    }

    fn type_kind(&self, ty: Type) -> TypeKind {
        match self.type_data(ty) {
            TypeData::Void => TypeKind::Void,
            TypeData::Int(_) => TypeKind::Integer,
            TypeData::Float(16) => TypeKind::Half,
            TypeData::Float(32) => TypeKind::Float,
            TypeData::Float(64) => TypeKind::Double,
            TypeData::Float(128) => TypeKind::FP128,
            TypeData::Float(_) => TypeKind::Float,
            TypeData::Ptr => TypeKind::Pointer,
            TypeData::Array(..) => TypeKind::Array,
            TypeData::Vector(..) => TypeKind::Vector,
            TypeData::Pair(..) | TypeData::Aggregate { .. } => TypeKind::Struct,
            TypeData::Func { .. } => TypeKind::Function,
        }
    }

    fn type_ptr(&self) -> Type {
        self.intern_type(TypeData::Ptr)
    }
    fn type_ptr_ext(&self, _address_space: AddressSpace) -> Type {
        self.intern_type(TypeData::Ptr)
    }

    fn element_type(&self, ty: Type) -> Type {
        match self.type_data(ty) {
            TypeData::Array(elem, _) | TypeData::Vector(elem, _) => elem,
            other => panic!("element_type called on non-aggregate type {other:?}"),
        }
    }

    fn vector_length(&self, ty: Type) -> usize {
        match self.type_data(ty) {
            TypeData::Vector(_, count) => count as usize,
            other => panic!("vector_length called on non-vector type {other:?}"),
        }
    }

    fn float_width(&self, ty: Type) -> usize {
        match self.type_data(ty) {
            TypeData::Float(bits) => bits as usize,
            other => panic!("float_width called on non-float type {other:?}"),
        }
    }

    fn int_width(&self, ty: Type) -> u64 {
        match self.type_data(ty) {
            TypeData::Int(bits) => bits as u64,
            other => panic!("int_width called on non-integer type {other:?}"),
        }
    }

    fn val_ty(&self, v: Self::Value) -> Type {
        v.ty()
    }
}

impl<'tcx> LayoutTypeCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn backend_type(&self, layout: TyAndLayout<'tcx>) -> Type {
        self.layout_backend_type(layout, false)
    }

    fn immediate_backend_type(&self, layout: TyAndLayout<'tcx>) -> Type {
        self.layout_backend_type(layout, true)
    }

    fn cast_backend_type(&self, ty: &CastTarget) -> Type {
        // Flatten the cast target into the sequence of registers it occupies. A single-register
        // cast maps to that register's scalar type; any multi-register cast (a small composite in
        // `x0:x1`, or an HFA in `v0..v3`) maps to an opaque aggregate addressed by byte offset, so
        // the value is a faithful byte image that the builder splits across the ABI registers using
        // the `CastTarget`'s own layout (a `Pair` would impose scalar-pair field offsets that need
        // not match the cast's, mishandling odd-width remainders).
        let mut regs: Vec<Reg> = ty.prefix.iter().copied().collect();
        let unit_size = ty.rest.unit.size.bytes().max(1);
        let total = ty.rest.total.bytes();
        for _ in 0..(total / unit_size) {
            regs.push(ty.rest.unit);
        }
        let rem = total % unit_size;
        if rem != 0 {
            regs.push(Reg { kind: RegKind::Integer, size: rustc_abi::Size::from_bytes(rem) });
        }
        match regs.len() {
            1 => self.reg_backend_type(&regs[0]),
            _ => {
                let size = ty.size(self);
                let align = ty.align(self);
                self.intern_type(TypeData::Aggregate { size: size.bytes(), align: align.bytes() })
            }
        }
    }

    fn fn_decl_backend_type(&self, _fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> Type {
        // The structural signature is unused by the baseline (the real ABI is taken from `FnAbi`);
        // a placeholder keeps the `BackendTypes::FunctionSignature` contract satisfied.
        let void = self.intern_type(TypeData::Void);
        self.intern_type(TypeData::Func { ret: void })
    }

    fn fn_ptr_backend_type(&self, _fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> Type {
        self.type_ptr()
    }

    fn reg_backend_type(&self, ty: &Reg) -> Type {
        match ty.kind {
            RegKind::Integer => self.intern_type(TypeData::Int(ty.size.bits() as u32)),
            RegKind::Float => self.intern_type(TypeData::Float(ty.size.bits() as u32)),
            RegKind::Vector { .. } => self
                .intern_type(TypeData::Aggregate { size: ty.size.bytes(), align: ty.size.bytes() }),
        }
    }

    fn is_backend_immediate(&self, layout: TyAndLayout<'tcx>) -> bool {
        matches!(
            layout.backend_repr,
            BackendRepr::Scalar(_) | BackendRepr::SimdVector { .. }
        )
    }

    fn is_backend_scalar_pair(&self, layout: TyAndLayout<'tcx>) -> bool {
        matches!(layout.backend_repr, BackendRepr::ScalarPair(..))
    }

    fn scalar_pair_element_backend_type(
        &self,
        layout: TyAndLayout<'tcx>,
        index: usize,
        immediate: bool,
    ) -> Type {
        let scalar = match layout.backend_repr {
            BackendRepr::ScalarPair(a, b) => {
                if index == 0 {
                    a
                } else {
                    b
                }
            }
            _ => panic!("scalar_pair_element_backend_type on non-pair layout"),
        };
        self.scalar_backend_type(scalar, immediate)
    }
}

impl<'tcx> TypeMembershipCodegenMethods<'tcx> for CodegenCx<'tcx> {}
