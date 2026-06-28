//! The `Builder`: per-function machine-code construction.
//!
//! The baseline model keeps every SSA value in a stack slot (no register allocator). A `Builder`
//! appends [`Inst`]s to the current basic block of the [`FunctionBuild`] held by the context; when
//! the function is finished its blocks are concatenated (each prefixed with its label) into a
//! [`MachFunction`] and pushed to the module.

use std::ops::Deref;

use rustc_abi::{Align, BackendRepr, HasDataLayout, Scalar, Size, TargetDataLayout, WrappingRange};
use rustc_ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_codegen_ssa::common::{
    AtomicRmwBinOp, IntPredicate, RealPredicate, SynchronizationScope,
};
use rustc_codegen_ssa::mir::IntrinsicResult;
use rustc_codegen_ssa::mir::operand::{OperandRef, OperandValue};
use rustc_codegen_ssa::mir::place::{PlaceRef, PlaceValue};
use rustc_codegen_ssa::traits::{
    AbiBuilderMethods, ArgAbiBuilderMethods, AsmBuilderMethods, BackendTypes, BuilderMethods,
    ConstCodegenMethods, CoverageInfoBuilderMethods, DebugInfoBuilderMethods, InlineAsmOperandRef,
    IntrinsicCallBuilderMethods, LayoutTypeCodegenMethods, OverflowOp, StaticBuilderMethods,
};
use rustc_codegen_ssa::{MemFlags, RetagInfo};
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs;
use rustc_middle::mir::coverage::CoverageKind;
use rustc_middle::ty::layout::{
    FnAbiOf, FnAbiOfHelpers, HasTyCtxt, HasTypingEnv, LayoutOfHelpers, TyAndLayout,
};
use rustc_middle::ty::{self, AtomicOrdering, Instance, Ty, TyCtxt};
use rustc_span::{Span, sym};
use rustc_target::callconv::{ArgAbi, FnAbi, PassMode};
use rustc_target::spec::{HasTargetSpec, Target};

use crate::context::{BasicBlock, CodegenCx, Function, Type, TypeData, Value};
use crate::mach::func::MachFunction;
use crate::mach::inst::{
    AddSub, AtomicRmwOp, CondSel, DataProc1, DataProc2, DmbOption, FpOp1, FpOp2, Inst, Label,
    LogicOp, MemSize, MovKind, PairIndex, SymRef,
};
use crate::mach::reg::{
    Cond, FpSize, Gpr, OperandSize, Vreg, FP, LR, SP, V0, V16, V17, X0, X1, X2, X9, X10, X12, X13,
    ZR,
};

/// Bump allocator for a function's stack frame. Slots are assigned at increasing offsets above the
/// reserved outgoing-argument area; the frame size is rounded to 16 bytes at finalization.
#[derive(Default)]
pub struct FrameAlloc {
    /// Current high-water mark in bytes.
    size: u64,
}

impl FrameAlloc {
    /// Reserve `size` bytes at the given alignment, returning the slot's byte offset from `sp`.
    pub fn alloc(&mut self, size: u64, align: u64) -> u64 {
        let align = align.max(1);
        let offset = align_up(self.size, align);
        self.size = offset + size.max(1);
        offset
    }

    /// The 16-byte-aligned frame size.
    pub fn frame_size(&self) -> u64 {
        align_up(self.size, 16)
    }
}

fn align_up(value: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

/// State for the function currently being lowered.
pub struct FunctionBuild {
    pub func: Function,
    pub name: Box<str>,
    pub is_global: bool,
    /// Instruction list for each basic block, indexed by block id (which is also its [`Label`]).
    pub blocks: Vec<Vec<Inst>>,
    pub frame: FrameAlloc,
    /// Spilled location of each physical incoming parameter, indexed by physical param index.
    pub param_slots: Vec<Value>,
}

impl FunctionBuild {
    pub fn new(func: Function, name: Box<str>, is_global: bool) -> FunctionBuild {
        FunctionBuild {
            func,
            name,
            is_global,
            blocks: Vec::new(),
            frame: FrameAlloc::default(),
            param_slots: Vec::new(),
        }
    }

    /// Append a fresh empty basic block, returning its id.
    pub fn new_block(&mut self) -> BasicBlock {
        let id = self.blocks.len() as u32;
        self.blocks.push(Vec::new());
        BasicBlock(id)
    }

    /// Concatenate the blocks (each prefixed with its label) into a finished [`MachFunction`],
    /// wrapping the body in the AArch64 prologue/epilogue. Each `ret` in the body is expanded into
    /// the stack-teardown sequence followed by the return.
    pub fn finish(self) -> MachFunction {
        let frame = self.frame.frame_size();
        let mut f = MachFunction::new(self.name, self.is_global);

        // Prologue: save FP/LR, set up the frame pointer, reserve the local frame.
        f.push(Inst::LoadStorePair {
            load: false,
            size: OperandSize::S64,
            index: PairIndex::PreIndex,
            rt: FP,
            rt2: LR,
            rn: SP,
            offset: -16,
        });
        f.push(Inst::MovSp { size: OperandSize::S64, rd: FP, rn: SP });
        if frame > 0 {
            f.push(sub_sp(frame));
        }

        for (id, block) in self.blocks.into_iter().enumerate() {
            f.push(Inst::Label(id as Label));
            for inst in block {
                if let Inst::Ret { .. } = inst {
                    // Epilogue before every return.
                    if frame > 0 {
                        f.push(add_sp(frame));
                    }
                    f.push(Inst::LoadStorePair {
                        load: true,
                        size: OperandSize::S64,
                        index: PairIndex::PostIndex,
                        rt: FP,
                        rt2: LR,
                        rn: SP,
                        offset: 16,
                    });
                }
                f.push(inst);
            }
        }
        f
    }
}

/// `sub sp, sp, #frame` (frame assumed to fit in a 12-bit immediate for now).
fn sub_sp(frame: u64) -> Inst {
    Inst::AddSubImm {
        op: AddSub::Sub,
        size: OperandSize::S64,
        set_flags: false,
        rd: SP,
        rn: SP,
        imm12: frame as u16,
        shift12: false,
    }
}

/// `add sp, sp, #frame`.
fn add_sp(frame: u64) -> Inst {
    Inst::AddSubImm {
        op: AddSub::Add,
        size: OperandSize::S64,
        set_flags: false,
        rd: SP,
        rn: SP,
        imm12: frame as u16,
        shift12: false,
    }
}

/// A builder positioned at the end of a basic block.
pub struct Builder<'a, 'tcx> {
    pub cx: &'a CodegenCx<'tcx>,
    /// The block instructions are currently appended to.
    pub block: BasicBlock,
}

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Append an instruction to the current block.
    pub fn emit(&mut self, inst: Inst) {
        let mut cur = self.cx.cur_fn.borrow_mut();
        let fb = cur.as_mut().expect("no function is currently being built");
        fb.blocks[self.block.0 as usize].push(inst);
    }

    /// Reserve a stack slot of the given size/alignment, returning its `sp`-relative offset.
    pub fn alloc_slot(&self, size: u64, align: u64) -> u64 {
        let mut cur = self.cx.cur_fn.borrow_mut();
        let fb = cur.as_mut().expect("no function is currently being built");
        fb.frame.alloc(size, align)
    }
}

impl<'a, 'tcx> Deref for Builder<'a, 'tcx> {
    type Target = CodegenCx<'tcx>;

    fn deref(&self) -> &Self::Target {
        self.cx
    }
}

impl<'a, 'tcx> BackendTypes for Builder<'a, 'tcx> {
    type Function = Function;
    type BasicBlock = BasicBlock;
    type Funclet = ();

    type Value = Value;
    type Type = Type;
    type FunctionSignature = Type;

    type DIScope = ();
    type DILocation = ();
    type DIVariable = ();
}

// The layout/abi helper traits delegate to the context so the blanket `LayoutOf`/`FnAbiOf` impls
// apply to the builder as well.
impl HasDataLayout for Builder<'_, '_> {
    fn data_layout(&self) -> &TargetDataLayout {
        self.cx.data_layout()
    }
}
impl<'tcx> HasTyCtxt<'tcx> for Builder<'_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.cx.tcx
    }
}
impl HasTargetSpec for Builder<'_, '_> {
    fn target_spec(&self) -> &Target {
        &self.cx.tcx.sess.target
    }
}
impl<'tcx> HasTypingEnv<'tcx> for Builder<'_, 'tcx> {
    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        ty::TypingEnv::fully_monomorphized()
    }
}
impl<'tcx> LayoutOfHelpers<'tcx> for Builder<'_, 'tcx> {
    fn handle_layout_err(
        &self,
        err: rustc_middle::ty::layout::LayoutError<'tcx>,
        span: rustc_span::Span,
        ty: Ty<'tcx>,
    ) -> ! {
        self.cx.handle_layout_err(err, span, ty)
    }
}
impl<'tcx> FnAbiOfHelpers<'tcx> for Builder<'_, 'tcx> {
    fn handle_fn_abi_err(
        &self,
        err: rustc_middle::ty::layout::FnAbiError<'tcx>,
        span: rustc_span::Span,
        request: rustc_middle::ty::layout::FnAbiRequest<'tcx>,
    ) -> ! {
        self.cx.handle_fn_abi_err(err, span, request)
    }
}

/// Integer operand size for a backend type.
fn op_size(cx: &CodegenCx<'_>, ty: Type) -> OperandSize {
    match cx.type_data(ty) {
        // 128-bit integers would need a register pair (or libcalls); silently truncating them to a
        // single 64-bit register would miscompile, so fail loudly until they are implemented.
        // See the "Future work" section in `lib.rs` for the intended implementation strategy.
        TypeData::Int(bits) if bits > 64 => {
            todo!("{bits}-bit integer operations are not yet supported by the arm64 backend")
        }
        TypeData::Int(bits) => OperandSize::from_bits(bits as u64),
        _ => OperandSize::S64,
    }
}

/// Memory access width for a backend type.
fn mem_size(cx: &CodegenCx<'_>, ty: Type) -> MemSize {
    let (size, _) = cx.type_size_align(ty);
    match size {
        1 => MemSize::B,
        2 => MemSize::H,
        4 => MemSize::W,
        _ => MemSize::X,
    }
}

/// Whether a backend type is a floating-point scalar (passed/returned in the SIMD&FP registers).
fn type_is_float(cx: &CodegenCx<'_>, ty: Type) -> bool {
    matches!(cx.type_data(ty), TypeData::Float(_))
}

/// Floating-point operand size for a float backend type (`f32` -> single, otherwise double).
fn fp_size(cx: &CodegenCx<'_>, ty: Type) -> FpSize {
    match cx.type_data(ty) {
        TypeData::Float(32) => FpSize::S32,
        _ => FpSize::S64,
    }
}

/// The incoming physical register class of a parameter in the baseline ABI.
enum ParamReg {
    /// Integer/pointer argument in `x0..x7` (or the indirect-return pointer in `x8`).
    Gpr(Gpr),
    /// Floating-point argument in `v0..v7`.
    Fp(Vreg),
}

/// Assign one scalar parameter to the next integer or floating-point register.
fn push_scalar_param(
    cx: &CodegenCx<'_>,
    params: &mut Vec<(ParamReg, Type)>,
    ngrn: &mut u8,
    nsrn: &mut u8,
    ty: Type,
) {
    if type_is_float(cx, ty) {
        params.push((ParamReg::Fp(Vreg::from_encoding(*nsrn)), ty));
        *nsrn += 1;
    } else {
        params.push((ParamReg::Gpr(Gpr::from_encoding(*ngrn)), ty));
        *ngrn += 1;
    }
}

/// Compute the incoming register and backend type of each physical parameter, following the
/// baseline AAPCS64 split: integer/pointer arguments fill `x0..x7`, floating-point arguments fill
/// `v0..v7`, and an indirect-return (sret) pointer arrives in `x8`. Stack-passed arguments (once
/// the register banks are exhausted) are not yet handled.
fn build_param_list<'tcx>(
    cx: &CodegenCx<'tcx>,
    fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
) -> Vec<(ParamReg, Type)> {
    let mut params = Vec::new();
    let ptr = cx.intern_type(TypeData::Ptr);
    if fn_abi.ret.is_indirect() {
        // The indirect-return (sret) pointer arrives in x8.
        params.push((ParamReg::Gpr(Gpr::from_encoding(8)), ptr));
    }
    let mut ngrn: u8 = 0;
    let mut nsrn: u8 = 0;
    for arg in fn_abi.args.iter() {
        match arg.mode {
            PassMode::Ignore => {}
            PassMode::Direct(_) => {
                let ty = cx.immediate_backend_type(arg.layout);
                push_scalar_param(cx, &mut params, &mut ngrn, &mut nsrn, ty);
            }
            PassMode::Pair(..) => {
                let a = cx.scalar_pair_element_backend_type(arg.layout, 0, true);
                let b = cx.scalar_pair_element_backend_type(arg.layout, 1, true);
                push_scalar_param(cx, &mut params, &mut ngrn, &mut nsrn, a);
                push_scalar_param(cx, &mut params, &mut ngrn, &mut nsrn, b);
            }
            PassMode::Indirect { .. } => {
                params.push((ParamReg::Gpr(Gpr::from_encoding(ngrn)), ptr));
                ngrn += 1;
            }
            PassMode::Cast { .. } => {
                params.push((ParamReg::Gpr(Gpr::from_encoding(ngrn)), cx.intern_type(TypeData::Int(64))));
                ngrn += 1;
            }
        }
    }
    params
}

/// Spill each incoming parameter register into a frame slot at function entry and record the slots.
fn setup_params(cx: &CodegenCx<'_>, fb: &mut FunctionBuild, block: BasicBlock) {
    let params = match cx.cur_instance.get() {
        Some(instance) => {
            let fn_abi = cx.fn_abi_of_instance(instance, ty::List::empty());
            build_param_list(cx, fn_abi)
        }
        // The synthesized C `main` entry wrapper has no MIR instance. Where `main` is
        // `int main(int argc, char** argv)`, spill those two incoming registers so the generic
        // entry-wrapper builder can read them back via `get_param`.
        None if cx.tcx.sess.target.main_needs_argc_argv => vec![
            (ParamReg::Gpr(Gpr::from_encoding(0)), cx.intern_type(TypeData::Int(32))),
            (ParamReg::Gpr(Gpr::from_encoding(1)), cx.intern_type(TypeData::Ptr)),
        ],
        None => return,
    };
    for (reg, ty) in params {
        let (size, align) = cx.type_size_align(ty);
        let off = fb.frame.alloc(size, align) as u32;
        let inst = match reg {
            ParamReg::Gpr(reg) => Inst::LoadStoreUImm {
                load: false,
                signed: false,
                size: mem_size(cx, ty),
                rt: reg,
                rn: SP,
                offset: off,
            },
            ParamReg::Fp(reg) => Inst::LoadStoreFpUImm {
                load: false,
                size: fp_size(cx, ty),
                rt: reg,
                rn: SP,
                offset: off,
            },
        };
        fb.blocks[block.0 as usize].push(inst);
        fb.param_slots.push(Value::Slot { off, ty });
    }
}

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Load a value into `reg`, emitting whatever is needed (immediate move, frame load, or address
    /// formation).
    fn materialize(&mut self, val: Value, reg: Gpr) {
        match val {
            Value::Const { bits, ty } => self.load_imm(reg, bits, op_size(self.cx, ty)),
            Value::Undef { ty } => self.load_imm(reg, 0, op_size(self.cx, ty)),
            Value::Slot { off, ty } => {
                self.emit(Inst::LoadStoreUImm {
                    load: true,
                    signed: false,
                    size: mem_size(self.cx, ty),
                    rt: reg,
                    rn: SP,
                    offset: off,
                });
            }
            Value::Sym { sym, offset, ty: _ } => {
                let symref = SymRef { name: self.cx.sym_name(sym), addend: offset };
                self.emit(Inst::Adrp { rd: reg, sym: symref.clone() });
                self.emit(Inst::AddLo { rd: reg, rn: reg, sym: symref });
            }
        }
    }

    /// Emit a `movz`/`movk` sequence loading the low 64 bits of `bits` into `reg`.
    fn load_imm(&mut self, reg: Gpr, bits: u128, size: OperandSize) {
        // For a 32-bit (`W`) destination only the low 32 bits are meaningful, and `movk` only
        // permits shifts of 0 or 16 (the `hw` field is 1 bit); emitting a shift of 32/48 on a `W`
        // register is an illegal encoding that faults as SIGILL at runtime. Mask the value to the
        // destination width and restrict the move-wide chunks accordingly.
        let (mask, shifts): (u64, &[u8]) = match size {
            OperandSize::S32 => (0xffff_ffff, &[16]),
            OperandSize::S64 => (u64::MAX, &[16, 32, 48]),
        };
        let bits = bits as u64 & mask;
        self.emit(Inst::MovWide {
            kind: MovKind::Zero,
            size,
            rd: reg,
            imm16: (bits & 0xffff) as u16,
            shift: 0,
        });
        for &shift in shifts {
            let chunk = ((bits >> shift) & 0xffff) as u16;
            if chunk != 0 {
                self.emit(Inst::MovWide { kind: MovKind::Keep, size, rd: reg, imm16: chunk, shift });
            }
        }
    }

    /// Store `reg` into a fresh frame slot and return a [`Value::Slot`] referring to it.
    fn spill(&mut self, reg: Gpr, ty: Type) -> Value {
        let (size, align) = self.cx.type_size_align(ty);
        let off = self.alloc_slot(size, align) as u32;
        self.emit(Inst::LoadStoreUImm {
            load: false,
            signed: false,
            size: mem_size(self.cx, ty),
            rt: reg,
            rn: SP,
            offset: off,
        });
        Value::Slot { off, ty }
    }

    /// Population count (`ctpop`) via the classic SWAR algorithm. The input is zero-extended to 64
    /// bits so the unused high bits do not contribute; the result is the count as a 64-bit value
    /// (the caller narrows it to the intrinsic's result type).
    fn emit_ctpop(&mut self, v: Value) -> Value {
        let i64t = self.cx.intern_type(TypeData::Int(64));
        let x0 = self.intcast(v, i64t, false);
        let m1 = self.cx.const_uint(i64t, 0x5555_5555_5555_5555);
        let m2 = self.cx.const_uint(i64t, 0x3333_3333_3333_3333);
        let m4 = self.cx.const_uint(i64t, 0x0f0f_0f0f_0f0f_0f0f);
        let h01 = self.cx.const_uint(i64t, 0x0101_0101_0101_0101);
        let s1 = self.cx.const_uint(i64t, 1);
        let s2 = self.cx.const_uint(i64t, 2);
        let s4 = self.cx.const_uint(i64t, 4);
        let s56 = self.cx.const_uint(i64t, 56);
        // x -= (x >> 1) & 0x5555...
        let t = self.lshr(x0, s1);
        let t = self.and(t, m1);
        let x = self.sub(x0, t);
        // x = (x & 0x3333...) + ((x >> 2) & 0x3333...)
        let lo = self.and(x, m2);
        let t = self.lshr(x, s2);
        let hi = self.and(t, m2);
        let x = self.add(lo, hi);
        // x = (x + (x >> 4)) & 0x0f0f...
        let t = self.lshr(x, s4);
        let s = self.add(x, t);
        let x = self.and(s, m4);
        // count = (x * 0x0101...) >> 56
        let prod = self.mul(x, h01);
        self.lshr(prod, s56)
    }

    /// Count leading zeros (`ctlz`). The input is zero-extended to 64 bits and `clz` is taken; the
    /// extra `64 - N` high zero bits introduced by the extension are then subtracted off. `ctlz(0)`
    /// is `N`, which falls out correctly (`clz` of `0` is 64). Returned as a 64-bit value for the
    /// caller to narrow.
    fn emit_ctlz(&mut self, v: Value) -> Value {
        let n = match self.cx.type_data(v.ty()) {
            TypeData::Int(b) => b as u64,
            _ => 64,
        };
        let i64t = self.cx.intern_type(TypeData::Int(64));
        let x = self.intcast(v, i64t, false);
        self.materialize(x, X9);
        self.emit(Inst::DataProc1 { op: DataProc1::Clz, size: OperandSize::S64, rd: X9, rn: X9 });
        let clz = self.spill(X9, i64t);
        if n < 64 {
            let adjust = self.cx.const_uint(i64t, 64 - n);
            self.sub(clz, adjust)
        } else {
            clz
        }
    }

    /// Count trailing zeros (`cttz`/`cttz_nonzero`) as `clz(rbit(x))`. A guard bit set at position
    /// `N` makes `cttz(0) == N` (it cannot affect any lower set bit, so non-zero inputs are
    /// unchanged). Returned as a 64-bit value for the caller to narrow.
    fn emit_cttz(&mut self, v: Value) -> Value {
        let n = match self.cx.type_data(v.ty()) {
            TypeData::Int(b) => b as u64,
            _ => 64,
        };
        let i64t = self.cx.intern_type(TypeData::Int(64));
        let mut x = self.intcast(v, i64t, false);
        if n < 64 {
            let guard = self.cx.const_uint(i64t, 1u64 << n);
            x = self.or(x, guard);
        }
        self.materialize(x, X9);
        self.emit(Inst::DataProc1 { op: DataProc1::Rbit, size: OperandSize::S64, rd: X9, rn: X9 });
        self.emit(Inst::DataProc1 { op: DataProc1::Clz, size: OperandSize::S64, rd: X9, rn: X9 });
        self.spill(X9, i64t)
    }

    /// Saturating add/sub for an integer of width `n`. Operands are widened to 64 bits (sign- or
    /// zero-extended per `signed`) and the wrapping result is computed there. For `n < 64` that
    /// widened result cannot overflow 64 bits, so clamping into the `n`-bit range is exact; at
    /// `n == 64` the 64-bit overflow is detected directly (carry/borrow for unsigned, operand/result
    /// sign logic for signed) and the result is replaced with the saturation bound.
    fn emit_saturating(
        &mut self,
        a: Value,
        b: Value,
        is_add: bool,
        signed: bool,
        result_ty: Type,
    ) -> Value {
        let n = match self.cx.type_data(result_ty) {
            TypeData::Int(bits) => bits,
            _ => 64,
        };
        let i64t = self.cx.intern_type(TypeData::Int(64));
        let a64 = self.intcast(a, i64t, signed);
        let b64 = self.intcast(b, i64t, signed);
        let s = if is_add { self.add(a64, b64) } else { self.sub(a64, b64) };
        let zero = self.cx.const_uint(i64t, 0);

        let clamped = if signed {
            let max = self.cx.const_uint(i64t, ((1i128 << (n - 1)) - 1) as u64);
            let min = self.cx.const_uint(i64t, (-(1i128 << (n - 1))) as i64 as u64);
            if n < 64 {
                let too_high = self.icmp(IntPredicate::IntSGT, s, max);
                let s = self.select(too_high, max, s);
                let too_low = self.icmp(IntPredicate::IntSLT, s, min);
                self.select(too_low, min, s)
            } else {
                // Signed overflow: add -> operands share a sign that the result doesn't; sub ->
                // operands differ in sign and the result's sign differs from `a`. Both reduce to a
                // sign bit set in the AND of two xors.
                let axs = self.xor(a64, s);
                let term2 = if is_add { self.xor(b64, s) } else { self.xor(a64, b64) };
                let ovf_bits = self.and(axs, term2);
                let ovf = self.icmp(IntPredicate::IntSLT, ovf_bits, zero);
                let a_neg = self.icmp(IntPredicate::IntSLT, a64, zero);
                let bound = self.select(a_neg, min, max);
                self.select(ovf, bound, s)
            }
        } else if is_add {
            if n < 64 {
                let max = self.cx.const_uint(i64t, ((1u128 << n) - 1) as u64);
                let too_high = self.icmp(IntPredicate::IntUGT, s, max);
                self.select(too_high, max, s)
            } else {
                // Carry: the 64-bit sum wrapped iff it is now less than one of the operands.
                let max = self.cx.const_uint(i64t, u64::MAX);
                let ovf = self.icmp(IntPredicate::IntULT, s, a64);
                self.select(ovf, max, s)
            }
        } else {
            // Unsigned subtract saturates to 0 on borrow (`a < b`); valid for every width.
            let borrow = self.icmp(IntPredicate::IntULT, a64, b64);
            self.select(borrow, zero, s)
        };
        self.trunc(clamped, result_ty)
    }

    /// Load a floating-point value into the SIMD&FP register `vreg`.
    fn materialize_fp(&mut self, val: Value, vreg: Vreg) {        let ty = val.ty();
        let size = fp_size(self.cx, ty);
        match val {
            Value::Slot { off, .. } => {
                self.emit(Inst::LoadStoreFpUImm { load: true, size, rt: vreg, rn: SP, offset: off });
            }
            Value::Const { bits, .. } => {
                // Load the raw IEEE-754 bit pattern into a scratch GPR, then move it across.
                let gpr_size = match size {
                    FpSize::S32 => OperandSize::S32,
                    FpSize::S64 => OperandSize::S64,
                };
                self.load_imm(X9, bits, gpr_size);
                self.emit(Inst::FmovFromGpr { size, rd: vreg, rn: X9 });
            }
            Value::Undef { .. } => {
                self.emit(Inst::FmovFromGpr { size, rd: vreg, rn: ZR });
            }
            Value::Sym { .. } => {
                // A symbol address is never a floating-point value; treat as zero defensively.
                self.emit(Inst::FmovFromGpr { size, rd: vreg, rn: ZR });
            }
        }
    }

    /// Store the SIMD&FP register `vreg` into a fresh frame slot, returning a [`Value::Slot`].
    fn spill_fp(&mut self, vreg: Vreg, ty: Type) -> Value {
        let (size, align) = self.cx.type_size_align(ty);
        let off = self.alloc_slot(size, align) as u32;
        self.emit(Inst::LoadStoreFpUImm {
            load: false,
            size: fp_size(self.cx, ty),
            rt: vreg,
            rn: SP,
            offset: off,
        });
        Value::Slot { off, ty }
    }

    /// Emit a two-operand FP op: materialize both operands into scratch FP registers, compute into
    /// `v16`, and spill.
    fn fp_alu(&mut self, op: FpOp2, lhs: Value, rhs: Value) -> Value {
        let ty = lhs.ty();
        let size = fp_size(self.cx, ty);
        self.materialize_fp(lhs, V16);
        self.materialize_fp(rhs, V17);
        self.emit(Inst::FpDataProc2 { op, size, rd: V16, rn: V16, rm: V17 });
        self.spill_fp(V16, ty)
    }

    /// Emit a register-register ALU op: materialize both operands, compute into `x9`, spill.
    fn alu_rrr(
        &mut self,
        mk: impl Fn(OperandSize, Gpr, Gpr, Gpr) -> Inst,
        lhs: Value,
        rhs: Value,
    ) -> Value {
        let ty = lhs.ty();
        let size = op_size(self.cx, ty);
        self.materialize(lhs, X9);
        self.materialize(rhs, X10);
        self.emit(mk(size, X9, X9, X10));
        self.spill(X9, ty)
    }

    /// Emit `cmp`/`cset` for an integer comparison, returning an `i1` value.
    fn emit_icmp(&mut self, cond: Cond, lhs: Value, rhs: Value) -> Value {
        let size = op_size(self.cx, lhs.ty());
        self.materialize(lhs, X9);
        self.materialize(rhs, X10);
        self.emit(Inst::AddSubReg {
            op: AddSub::Sub,
            size,
            set_flags: true,
            rd: ZR,
            rn: X9,
            rm: X10,
            amount: 0,
        });
        // cset x9, cond  ==  csinc x9, xzr, xzr, invert(cond)
        self.emit(Inst::CondSel {
            op: CondSel::Csinc,
            size: OperandSize::S32,
            rd: X9,
            rn: ZR,
            rm: ZR,
            cond: cond.invert(),
        });
        self.spill(X9, self.cx.intern_type(TypeData::Int(1)))
    }
}

fn int_pred_to_cond(pred: IntPredicate) -> Cond {
    match pred {
        IntPredicate::IntEQ => Cond::Eq,
        IntPredicate::IntNE => Cond::Ne,
        IntPredicate::IntUGT => Cond::Hi,
        IntPredicate::IntUGE => Cond::Hs,
        IntPredicate::IntULT => Cond::Lo,
        IntPredicate::IntULE => Cond::Ls,
        IntPredicate::IntSGT => Cond::Gt,
        IntPredicate::IntSGE => Cond::Ge,
        IntPredicate::IntSLT => Cond::Lt,
        IntPredicate::IntSLE => Cond::Le,
    }
}

/// AArch64 condition for a floating-point predicate after `fcmp` (which sets NZCV so that an
/// unordered result, i.e. a NaN operand, has C=1, V=1). These are the single-condition mappings;
/// Rust's surface float comparisons only emit `OEQ`/`OGT`/`OGE`/`OLT`/`OLE`/`UNE`.
fn real_pred_to_cond(pred: RealPredicate) -> Cond {
    match pred {
        RealPredicate::RealOEQ => Cond::Eq,
        RealPredicate::RealOGT => Cond::Gt,
        RealPredicate::RealOGE => Cond::Ge,
        RealPredicate::RealOLT => Cond::Mi,
        RealPredicate::RealOLE => Cond::Ls,
        RealPredicate::RealUNE => Cond::Ne,
        RealPredicate::RealUGT => Cond::Hi,
        RealPredicate::RealUGE => Cond::Pl,
        RealPredicate::RealULT => Cond::Lt,
        RealPredicate::RealULE => Cond::Le,
        RealPredicate::RealORD => Cond::Vc,
        RealPredicate::RealUNO => Cond::Vs,
        other => todo!("rustc_codegen_arm64: float predicate {other:?}"),
    }
}

/// Memory access width for an atomic of the given byte size.
fn mem_size_from_bytes(bytes: u64) -> MemSize {
    match bytes {
        1 => MemSize::B,
        2 => MemSize::H,
        4 => MemSize::W,
        _ => MemSize::X,
    }
}

/// Whether an ordering implies an acquire barrier (sets the LSE `A` bit / selects a load-acquire).
fn order_acquire(order: AtomicOrdering) -> bool {
    matches!(order, AtomicOrdering::Acquire | AtomicOrdering::AcqRel | AtomicOrdering::SeqCst)
}

/// Whether an ordering implies a release barrier (sets the LSE `R` bit / selects a store-release).
fn order_release(order: AtomicOrdering) -> bool {
    matches!(order, AtomicOrdering::Release | AtomicOrdering::AcqRel | AtomicOrdering::SeqCst)
}

impl<'a, 'tcx> CoverageInfoBuilderMethods<'tcx> for Builder<'a, 'tcx> {
    fn add_coverage(&mut self, _instance: Instance<'tcx>, _kind: &CoverageKind) {}
}

impl<'a, 'tcx> DebugInfoBuilderMethods<'tcx> for Builder<'a, 'tcx> {
    fn dbg_var_addr(
        &mut self,
        _dbg_var: (),
        _dbg_loc: (),
        _variable_alloca: Value,
        _direct_offset: Size,
        _indirect_offsets: &[Size],
        _fragment: &Option<std::ops::Range<Size>>,
    ) {
    }
    fn dbg_var_value(
        &mut self,
        _dbg_var: (),
        _dbg_loc: (),
        _value: Value,
        _direct_offset: Size,
        _indirect_offsets: &[Size],
        _fragment: &Option<std::ops::Range<Size>>,
    ) {
    }
    fn set_dbg_loc(&mut self, _dbg_loc: ()) {}
    fn clear_dbg_loc(&mut self) {}
    fn insert_reference_to_gdb_debug_scripts_section_global(&mut self) {}
    fn set_var_name(&mut self, _value: Value, _name: &str) {}
}

impl<'a, 'tcx> AbiBuilderMethods for Builder<'a, 'tcx> {
    fn get_param(&mut self, index: usize) -> Value {
        self.cx.cur_fn.borrow().as_ref().expect("no function being built").param_slots[index]
    }
}

impl<'a, 'tcx> ArgAbiBuilderMethods<'tcx> for Builder<'a, 'tcx> {
    fn store_fn_arg(
        &mut self,
        arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        idx: &mut usize,
        dst: PlaceRef<'tcx, Value>,
    ) {
        match arg_abi.mode {
            PassMode::Ignore => {}
            PassMode::Direct(_) => {
                let val = self.get_param(*idx);
                *idx += 1;
                OperandValue::Immediate(val).store(self, dst);
            }
            PassMode::Pair(..) => {
                let a = self.get_param(*idx);
                let b = self.get_param(*idx + 1);
                *idx += 2;
                OperandValue::Pair(a, b).store(self, dst);
            }
            PassMode::Indirect { .. } => {
                // The argument is passed by reference; treat the incoming pointer as the value.
                let ptr = self.get_param(*idx);
                *idx += 1;
                OperandValue::Ref(rustc_codegen_ssa::mir::place::PlaceValue::new_sized(
                    ptr,
                    dst.val.align,
                ))
                .store(self, dst);
            }
            PassMode::Cast { .. } => {
                let val = self.get_param(*idx);
                *idx += 1;
                OperandValue::Immediate(val).store(self, dst);
            }
        }
    }
    fn store_arg(
        &mut self,
        _arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        val: Value,
        dst: PlaceRef<'tcx, Value>,
    ) {
        OperandValue::Immediate(val).store(self, dst);
    }
}

impl<'a, 'tcx> IntrinsicCallBuilderMethods<'tcx> for Builder<'a, 'tcx> {
    fn codegen_intrinsic_call(
        &mut self,
        instance: Instance<'tcx>,
        args: &[OperandRef<'tcx, Value>],
        result_layout: TyAndLayout<'tcx>,
        _result_place: Option<PlaceValue<Value>>,
        _span: Span,
    ) -> IntrinsicResult<'tcx, Value> {
        let name = self.cx.tcx.item_name(instance.def_id());
        match name {
            // `black_box` is an optimization barrier; with no optimizer it is the identity.
            sym::black_box => IntrinsicResult::Operand(args[0].val),
            // Population count: a SWAR sequence, narrowed to the `u32` result type.
            sym::ctpop => {
                let count = self.emit_ctpop(args[0].immediate());
                let result_ty = self.cx.immediate_backend_type(result_layout);
                let count = self.intcast(count, result_ty, false);
                IntrinsicResult::Operand(OperandValue::Immediate(count))
            }
            // Count leading zeros (`ctlz_nonzero` shares the lowering; `clz` handles zero anyway).
            sym::ctlz | sym::ctlz_nonzero => {
                let count = self.emit_ctlz(args[0].immediate());
                let result_ty = self.cx.immediate_backend_type(result_layout);
                let count = self.intcast(count, result_ty, false);
                IntrinsicResult::Operand(OperandValue::Immediate(count))
            }
            // Count trailing zeros, as `clz(rbit(x))` (the `_nonzero` form shares the lowering).
            sym::cttz | sym::cttz_nonzero => {
                let count = self.emit_cttz(args[0].immediate());
                let result_ty = self.cx.immediate_backend_type(result_layout);
                let count = self.intcast(count, result_ty, false);
                IntrinsicResult::Operand(OperandValue::Immediate(count))
            }
            // Saturating add/sub, clamped to the integer type's range.
            sym::saturating_add | sym::saturating_sub => {
                let is_add = name == sym::saturating_add;
                let signed = matches!(result_layout.ty.kind(), ty::Int(_));
                let result_ty = self.cx.immediate_backend_type(result_layout);
                let r = self.emit_saturating(
                    args[0].immediate(),
                    args[1].immediate(),
                    is_add,
                    signed,
                    result_ty,
                );
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            // Everything else falls back to the intrinsic's MIR body (if it has one).
            _ => IntrinsicResult::Fallback(instance),
        }
    }
    fn codegen_llvm_intrinsic_call(
        &mut self,
        _instance: Instance<'tcx>,
        _args: &[OperandRef<'tcx, Value>],
        _is_cleanup: bool,
    ) -> Value {
        todo!("rustc_codegen_arm64: codegen_llvm_intrinsic_call")
    }
    fn abort(&mut self) {
        self.emit(Inst::Brk { imm16: 1 });
    }
    fn assume(&mut self, _val: Value) {}
    fn expect(&mut self, cond: Value, _expected: bool) -> Value {
        cond
    }
    fn type_checked_load(&mut self, _llvtable: Value, _offset: u64, _typeid: &[u8]) -> Value {
        todo!("rustc_codegen_arm64: type_checked_load")
    }
    fn va_start(&mut self, _val: Value) {
        todo!("rustc_codegen_arm64: va_start")
    }
    fn retag_mem(&mut self, _place: Value, _info: &RetagInfo<Value>) {}
    fn retag_reg(&mut self, ptr: Value, _info: &RetagInfo<Value>) -> Value {
        ptr
    }
}

impl<'a, 'tcx> AsmBuilderMethods<'tcx> for Builder<'a, 'tcx> {
    fn codegen_inline_asm(
        &mut self,
        _template: &[InlineAsmTemplatePiece],
        _operands: &[InlineAsmOperandRef<'tcx, Self>],
        _options: InlineAsmOptions,
        _line_spans: &[Span],
        _instance: Instance<'_>,
        _dest: Option<BasicBlock>,
        _catch_funclet: Option<(BasicBlock, Option<&()>)>,
    ) {
        todo!("rustc_codegen_arm64: codegen_inline_asm")
    }
}

impl<'a, 'tcx> StaticBuilderMethods for Builder<'a, 'tcx> {
    fn get_static(&mut self, def_id: rustc_hir::def_id::DefId) -> Value {
        let sym = self.cx.get_static_sym(def_id);
        Value::Sym { sym, offset: 0, ty: self.ptr_ty() }
    }
}

impl<'a, 'tcx> Builder<'a, 'tcx> {
    fn ptr_ty(&self) -> Type {
        self.cx.intern_type(TypeData::Ptr)
    }

    /// Reinterpret a value as `dest_ty` without changing its bits (same-size casts).
    fn retype(&self, val: Value, dest_ty: Type) -> Value {
        match val {
            Value::Const { bits, .. } => Value::Const { bits, ty: dest_ty },
            Value::Slot { off, .. } => Value::Slot { off, ty: dest_ty },
            Value::Sym { sym, offset, .. } => Value::Sym { sym, offset, ty: dest_ty },
            Value::Undef { .. } => Value::Undef { ty: dest_ty },
        }
    }
}

impl<'a, 'tcx> BuilderMethods<'a, 'tcx> for Builder<'a, 'tcx> {
    type CodegenCx = CodegenCx<'tcx>;

    fn build(cx: &'a CodegenCx<'tcx>, llbb: BasicBlock) -> Self {
        Builder { cx, block: llbb }
    }

    fn cx(&self) -> &CodegenCx<'tcx> {
        self.cx
    }
    fn llbb(&self) -> BasicBlock {
        self.block
    }
    fn set_span(&mut self, _span: Span) {}

    fn append_block(cx: &'a CodegenCx<'tcx>, llfn: Function, _name: &str) -> BasicBlock {
        if cx.cur_fn.borrow().is_none() {
            let (sym, is_global) = {
                let funcs = cx.functions.borrow();
                let decl = &funcs[llfn.0 as usize];
                (decl.sym, decl.is_global)
            };
            let name = cx.sym_name(sym);
            let mut fb = FunctionBuild::new(llfn, name, is_global);
            let entry = fb.new_block();
            setup_params(cx, &mut fb, entry);
            *cx.cur_fn.borrow_mut() = Some(fb);
            entry
        } else {
            cx.cur_fn.borrow_mut().as_mut().unwrap().new_block()
        }
    }

    fn append_sibling_block(&mut self, _name: &str) -> BasicBlock {
        self.cx.cur_fn.borrow_mut().as_mut().expect("no function being built").new_block()
    }

    fn switch_to_block(&mut self, llbb: BasicBlock) {
        self.block = llbb;
    }

    fn ret_void(&mut self) {
        self.emit(Inst::Ret { rn: LR });
    }
    fn ret(&mut self, v: Value) {
        if let TypeData::Pair(..) = self.cx.type_data(v.ty()) {
            // `PassMode::Pair`: return field 0 in x0/v0 and field 1 in x1/v1 (per field class).
            let fields = [self.extract_value(v, 0), self.extract_value(v, 1)];
            let mut ngrn: u8 = 0;
            let mut nsrn: u8 = 0;
            for f in fields {
                if type_is_float(self.cx, f.ty()) {
                    self.materialize_fp(f, Vreg::from_encoding(nsrn));
                    nsrn += 1;
                } else {
                    self.materialize(f, Gpr::from_encoding(ngrn));
                    ngrn += 1;
                }
            }
        } else if type_is_float(self.cx, v.ty()) {
            self.materialize_fp(v, V0);
        } else {
            self.materialize(v, X0);
        }
        self.emit(Inst::Ret { rn: LR });
    }
    fn br(&mut self, dest: BasicBlock) {
        self.emit(Inst::B { target: dest.0 as Label });
    }
    fn cond_br(&mut self, cond: Value, then_llbb: BasicBlock, else_llbb: BasicBlock) {
        self.materialize(cond, X9);
        self.emit(Inst::CbNz {
            nonzero: true,
            size: OperandSize::S32,
            rt: X9,
            target: then_llbb.0 as Label,
        });
        self.emit(Inst::B { target: else_llbb.0 as Label });
    }
    fn switch(
        &mut self,
        v: Value,
        else_llbb: BasicBlock,
        cases: impl ExactSizeIterator<Item = (u128, BasicBlock)>,
    ) {
        self.materialize(v, X9);
        let size = op_size(self.cx, v.ty());
        for (case, bb) in cases {
            self.load_imm(X10, case, size);
            self.emit(Inst::AddSubReg {
                op: AddSub::Sub,
                size,
                set_flags: true,
                rd: ZR,
                rn: X9,
                rm: X10,
                amount: 0,
            });
            self.emit(Inst::BCond { cond: Cond::Eq, target: bb.0 as Label });
        }
        self.emit(Inst::B { target: else_llbb.0 as Label });
    }
    fn invoke(
        &mut self,
        _llty: Type,
        _fn_attrs: Option<&CodegenFnAttrs>,
        fn_abi: Option<&FnAbi<'tcx, Ty<'tcx>>>,
        llfn: Value,
        args: &[Value],
        then: BasicBlock,
        _catch: BasicBlock,
        _funclet: Option<&()>,
        _instance: Option<Instance<'tcx>>,
    ) -> Value {
        // Baseline model: emit the call and fall through to the normal successor. The unwind edge
        // (`_catch`) is not wired up; an unwind through this call aborts (panic=abort semantics).
        let ret = self.emit_call_core(fn_abi, llfn, args);
        self.br(then);
        ret
    }
    fn unreachable(&mut self) {
        self.emit(Inst::Brk { imm16: 1 });
    }

    fn add(&mut self, lhs: Value, rhs: Value) -> Value {
        self.alu_rrr(
            |size, rd, rn, rm| Inst::AddSubReg {
                op: AddSub::Add,
                size,
                set_flags: false,
                rd,
                rn,
                rm,
                amount: 0,
            },
            lhs,
            rhs,
        )
    }
    fn sub(&mut self, lhs: Value, rhs: Value) -> Value {
        self.alu_rrr(
            |size, rd, rn, rm| Inst::AddSubReg {
                op: AddSub::Sub,
                size,
                set_flags: false,
                rd,
                rn,
                rm,
                amount: 0,
            },
            lhs,
            rhs,
        )
    }
    fn mul(&mut self, lhs: Value, rhs: Value) -> Value {
        self.alu_rrr(|size, rd, rn, rm| Inst::Madd { size, rd, rn, rm, ra: ZR }, lhs, rhs)
    }
    fn udiv(&mut self, lhs: Value, rhs: Value) -> Value {
        self.alu_rrr(
            |size, rd, rn, rm| Inst::DataProc2 { op: DataProc2::Udiv, size, rd, rn, rm },
            lhs,
            rhs,
        )
    }
    fn exactudiv(&mut self, lhs: Value, rhs: Value) -> Value {
        self.udiv(lhs, rhs)
    }
    fn sdiv(&mut self, lhs: Value, rhs: Value) -> Value {
        self.alu_rrr(
            |size, rd, rn, rm| Inst::DataProc2 { op: DataProc2::Sdiv, size, rd, rn, rm },
            lhs,
            rhs,
        )
    }
    fn exactsdiv(&mut self, lhs: Value, rhs: Value) -> Value {
        self.sdiv(lhs, rhs)
    }
    fn urem(&mut self, lhs: Value, rhs: Value) -> Value {
        self.rem(lhs, rhs, DataProc2::Udiv)
    }
    fn srem(&mut self, lhs: Value, rhs: Value) -> Value {
        self.rem(lhs, rhs, DataProc2::Sdiv)
    }
    fn and(&mut self, lhs: Value, rhs: Value) -> Value {
        self.alu_rrr(
            |size, rd, rn, rm| Inst::Logical { op: LogicOp::And, size, rd, rn, rm, amount: 0 },
            lhs,
            rhs,
        )
    }
    fn or(&mut self, lhs: Value, rhs: Value) -> Value {
        self.alu_rrr(
            |size, rd, rn, rm| Inst::Logical { op: LogicOp::Orr, size, rd, rn, rm, amount: 0 },
            lhs,
            rhs,
        )
    }
    fn xor(&mut self, lhs: Value, rhs: Value) -> Value {
        self.alu_rrr(
            |size, rd, rn, rm| Inst::Logical { op: LogicOp::Eor, size, rd, rn, rm, amount: 0 },
            lhs,
            rhs,
        )
    }
    fn shl(&mut self, lhs: Value, rhs: Value) -> Value {
        self.alu_rrr(
            |size, rd, rn, rm| Inst::DataProc2 { op: DataProc2::Lslv, size, rd, rn, rm },
            lhs,
            rhs,
        )
    }
    fn lshr(&mut self, lhs: Value, rhs: Value) -> Value {
        self.alu_rrr(
            |size, rd, rn, rm| Inst::DataProc2 { op: DataProc2::Lsrv, size, rd, rn, rm },
            lhs,
            rhs,
        )
    }
    fn ashr(&mut self, lhs: Value, rhs: Value) -> Value {
        self.alu_rrr(
            |size, rd, rn, rm| Inst::DataProc2 { op: DataProc2::Asrv, size, rd, rn, rm },
            lhs,
            rhs,
        )
    }
    fn neg(&mut self, v: Value) -> Value {
        let zero = Value::Const { bits: 0, ty: v.ty() };
        self.sub(zero, v)
    }
    fn fneg(&mut self, v: Value) -> Value {
        let ty = v.ty();
        let size = fp_size(self.cx, ty);
        self.materialize_fp(v, V16);
        self.emit(Inst::FpDataProc1 { op: FpOp1::Fneg, size, rd: V16, rn: V16 });
        self.spill_fp(V16, ty)
    }
    fn not(&mut self, v: Value) -> Value {
        // Complement within the value's own bit width. This matters for `bool` (`i1`): `!true` must
        // be `0`, but xoring with a full-width all-ones and truncating to the 1-byte slot would
        // leave `0xfe`, which reads as truthy. Masking to the type width keeps `!bool` in `{0, 1}`.
        let mask = match self.cx.type_data(v.ty()) {
            TypeData::Int(bits) if bits < 128 => (1u128 << bits) - 1,
            _ => u128::MAX,
        };
        let all_ones = Value::Const { bits: mask, ty: v.ty() };
        self.xor(v, all_ones)
    }

    fn fadd(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fp_alu(FpOp2::Fadd, lhs, rhs)
    }
    fn fadd_fast(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fadd(lhs, rhs)
    }
    fn fadd_algebraic(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fadd(lhs, rhs)
    }
    fn fsub(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fp_alu(FpOp2::Fsub, lhs, rhs)
    }
    fn fsub_fast(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fsub(lhs, rhs)
    }
    fn fsub_algebraic(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fsub(lhs, rhs)
    }
    fn fmul(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fp_alu(FpOp2::Fmul, lhs, rhs)
    }
    fn fmul_fast(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fmul(lhs, rhs)
    }
    fn fmul_algebraic(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fmul(lhs, rhs)
    }
    fn fdiv(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fp_alu(FpOp2::Fdiv, lhs, rhs)
    }
    fn fdiv_fast(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fdiv(lhs, rhs)
    }
    fn fdiv_algebraic(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fdiv(lhs, rhs)
    }
    fn frem(&mut self, _lhs: Value, _rhs: Value) -> Value {
        // `frem` has no hardware instruction; it requires a call to `fmod`/`fmodf`. Deferred.
        todo!("rustc_codegen_arm64: frem")
    }
    fn frem_fast(&mut self, lhs: Value, rhs: Value) -> Value {
        self.frem(lhs, rhs)
    }
    fn frem_algebraic(&mut self, lhs: Value, rhs: Value) -> Value {
        self.frem(lhs, rhs)
    }

    fn checked_binop(
        &mut self,
        oop: OverflowOp,
        ty: Ty<'tcx>,
        lhs: Value,
        rhs: Value,
    ) -> (Value, Value) {
        let signed = ty.is_signed();
        let val_ty = lhs.ty();
        if let OverflowOp::Mul = oop {
            return self.checked_mul(signed, val_ty, lhs, rhs);
        }
        let size = op_size(self.cx, lhs.ty());
        let bool_ty = self.cx.intern_type(TypeData::Int(1));
        self.materialize(lhs, X9);
        self.materialize(rhs, X10);
        let overflow_cond = match oop {
            OverflowOp::Add => {
                self.emit(Inst::AddSubReg {
                    op: AddSub::Add,
                    size,
                    set_flags: true,
                    rd: X9,
                    rn: X9,
                    rm: X10,
                    amount: 0,
                });
                if signed { Cond::Vs } else { Cond::Hs }
            }
            OverflowOp::Sub => {
                self.emit(Inst::AddSubReg {
                    op: AddSub::Sub,
                    size,
                    set_flags: true,
                    rd: X9,
                    rn: X9,
                    rm: X10,
                    amount: 0,
                });
                if signed { Cond::Vs } else { Cond::Lo }
            }
            OverflowOp::Mul => unreachable!("multiplication is handled by checked_mul above"),
        };
        // Read the overflow flag into a bool (cset wN, cond == csinc wN, wzr, wzr, invert(cond)).
        self.emit(Inst::CondSel {
            op: CondSel::Csinc,
            size: OperandSize::S32,
            rd: X11_HACK,
            rn: ZR,
            rm: ZR,
            cond: overflow_cond.invert(),
        });
        let result = self.spill(X9, val_ty);
        let overflow = self.spill(X11_HACK, bool_ty);
        (result, overflow)
    }

    fn from_immediate(&mut self, val: Value) -> Value {
        val
    }
    fn to_immediate_scalar(&mut self, val: Value, _scalar: Scalar) -> Value {
        val
    }

    fn alloca(&mut self, size: Size, align: Align) -> Value {
        let off = self.alloc_slot(size.bytes(), align.bytes());
        self.emit(Inst::AddSubImm {
            op: AddSub::Add,
            size: OperandSize::S64,
            set_flags: false,
            rd: X9,
            rn: SP,
            imm12: off as u16,
            shift12: false,
        });
        let ptr_ty = self.ptr_ty();
        self.spill(X9, ptr_ty)
    }
    fn alloca_with_ty(&mut self, layout: TyAndLayout<'tcx>) -> Value {
        self.alloca(layout.size, layout.align.abi)
    }

    fn load(&mut self, ty: Type, ptr: Value, _align: Align) -> Value {
        if let TypeData::Pair(..) = self.cx.type_data(ty) {
            return self.load_pair(ty, ptr);
        }
        self.materialize(ptr, X9);
        self.emit(Inst::LoadStoreUImm {
            load: true,
            signed: false,
            size: mem_size(self.cx, ty),
            rt: X10,
            rn: X9,
            offset: 0,
        });
        self.spill(X10, ty)
    }
    fn volatile_load(&mut self, ty: Type, ptr: Value) -> Value {
        self.load(ty, ptr, Align::ONE)
    }
    fn atomic_load(&mut self, ty: Type, ptr: Value, order: AtomicOrdering, size: Size) -> Value {
        let msize = mem_size_from_bytes(size.bytes());
        self.materialize(ptr, X9);
        if order_acquire(order) {
            self.emit(Inst::LoadAcq { size: msize, rt: X10, rn: X9 });
        } else {
            self.emit(Inst::LoadStoreUImm {
                load: true,
                signed: false,
                size: msize,
                rt: X10,
                rn: X9,
                offset: 0,
            });
        }
        self.spill(X10, ty)
    }
    fn load_operand(&mut self, place: PlaceRef<'tcx, Value>) -> OperandRef<'tcx, Value> {
        if place.layout.is_zst() {
            return OperandRef::zero_sized(place.layout);
        }
        let val = if self.cx.is_backend_immediate(place.layout) {
            let ty = self.cx.immediate_backend_type(place.layout);
            OperandValue::Immediate(self.load(ty, place.val.llval, place.val.align))
        } else if let BackendRepr::ScalarPair(a, b) = place.layout.backend_repr {
            // A scalar pair (e.g. a wide pointer or an enum variant with a niche): load each
            // component separately so the operand is a `Pair`. The generic codegen requires this
            // representation for scalar-pair layouts (a `Ref` here triggers a "non-pair" ICE).
            let b_offset = a.size(self.cx).align_to(b.align(self.cx).abi);
            let ty_a = self.cx.scalar_pair_element_backend_type(place.layout, 0, true);
            let ty_b = self.cx.scalar_pair_element_backend_type(place.layout, 1, true);
            let off = self.const_usize(b_offset.bytes());
            let ptr_b = self.inbounds_ptradd(place.val.llval, off);
            let a_val = self.load(ty_a, place.val.llval, place.val.align);
            let b_val = self.load(ty_b, ptr_b, place.val.align);
            OperandValue::Pair(a_val, b_val)
        } else {
            OperandValue::Ref(place.val)
        };
        OperandRef { val, layout: place.layout, move_annotation: None }
    }

    fn write_operand_repeatedly(
        &mut self,
        _elem: OperandRef<'tcx, Value>,
        _count: u64,
        _dest: PlaceRef<'tcx, Value>,
    ) {
        todo!("rustc_codegen_arm64: write_operand_repeatedly")
    }

    fn range_metadata(&mut self, _load: Value, _range: WrappingRange) {}
    fn nonnull_metadata(&mut self, _load: Value) {}

    fn store(&mut self, val: Value, ptr: Value, align: Align) -> Value {
        self.store_with_flags(val, ptr, align, MemFlags::empty())
    }
    fn store_with_flags(
        &mut self,
        val: Value,
        ptr: Value,
        _align: Align,
        _flags: MemFlags,
    ) -> Value {
        if let TypeData::Pair(..) = self.cx.type_data(val.ty()) {
            self.store_pair(val, ptr);
            return Value::Undef { ty: self.ptr_ty() };
        }
        self.materialize(ptr, X9);
        self.materialize(val, X10);
        self.emit(Inst::LoadStoreUImm {
            load: false,
            signed: false,
            size: mem_size(self.cx, val.ty()),
            rt: X10,
            rn: X9,
            offset: 0,
        });
        Value::Undef { ty: self.ptr_ty() }
    }
    fn atomic_store(&mut self, val: Value, ptr: Value, order: AtomicOrdering, size: Size) {
        let msize = mem_size_from_bytes(size.bytes());
        self.materialize(ptr, X9);
        self.materialize(val, X10);
        if order_release(order) {
            self.emit(Inst::StoreRel { size: msize, rt: X10, rn: X9 });
        } else {
            self.emit(Inst::LoadStoreUImm {
                load: false,
                signed: false,
                size: msize,
                rt: X10,
                rn: X9,
                offset: 0,
            });
        }
    }

    fn gep(&mut self, ty: Type, ptr: Value, indices: &[Value]) -> Value {
        assert_eq!(indices.len(), 1, "only single-index gep is supported in the baseline");
        let (elem_size, _) = self.cx.type_size_align(ty);
        self.materialize(ptr, X9);
        self.materialize(indices[0], X10);
        self.load_imm(X11_HACK, elem_size as u128, OperandSize::S64);
        // x9 = idx * elem_size + ptr
        self.emit(Inst::Madd {
            size: OperandSize::S64,
            rd: X9,
            rn: X10,
            rm: X11_HACK,
            ra: X9,
        });
        let ptr_ty = self.ptr_ty();
        self.spill(X9, ptr_ty)
    }
    fn inbounds_gep(&mut self, ty: Type, ptr: Value, indices: &[Value]) -> Value {
        self.gep(ty, ptr, indices)
    }

    fn trunc(&mut self, val: Value, dest_ty: Type) -> Value {
        // Little-endian: reading fewer low bytes truncates; just retype.
        self.retype(val, dest_ty)
    }
    fn sext(&mut self, val: Value, dest_ty: Type) -> Value {
        let src_bits = match self.cx.type_data(val.ty()) {
            TypeData::Int(b) => b,
            _ => 64,
        };
        self.materialize(val, X9);
        let shift = (64 - src_bits) as u128;
        if shift != 0 {
            self.load_imm(X10, shift, OperandSize::S64);
            self.emit(Inst::DataProc2 {
                op: DataProc2::Lslv,
                size: OperandSize::S64,
                rd: X9,
                rn: X9,
                rm: X10,
            });
            self.emit(Inst::DataProc2 {
                op: DataProc2::Asrv,
                size: OperandSize::S64,
                rd: X9,
                rn: X9,
                rm: X10,
            });
        }
        self.spill(X9, dest_ty)
    }
    fn zext(&mut self, val: Value, dest_ty: Type) -> Value {
        // Materializing a slot zero-extends (e.g. `ldrb`); re-spill at the wider type.
        self.materialize(val, X9);
        self.spill(X9, dest_ty)
    }
    fn fptoui_sat(&mut self, val: Value, dest_ty: Type) -> Value {
        // AArch64 `fcvtzu` already saturates out-of-range values and maps NaN to zero.
        self.fptoui(val, dest_ty)
    }
    fn fptosi_sat(&mut self, val: Value, dest_ty: Type) -> Value {
        // AArch64 `fcvtzs` already saturates out-of-range values and maps NaN to zero.
        self.fptosi(val, dest_ty)
    }
    fn fptoui(&mut self, val: Value, dest_ty: Type) -> Value {
        // FIXME: for destination widths < 32 bits the result saturates to the 32-bit range rather
        // than the narrow type's range; correct for i32/i64/u32/u64.
        let fp = fp_size(self.cx, val.ty());
        let int = op_size(self.cx, dest_ty);
        self.materialize_fp(val, V16);
        self.emit(Inst::FpToInt { signed: false, fp, int, rd: X9, rn: V16 });
        self.spill(X9, dest_ty)
    }
    fn fptosi(&mut self, val: Value, dest_ty: Type) -> Value {
        let fp = fp_size(self.cx, val.ty());
        let int = op_size(self.cx, dest_ty);
        self.materialize_fp(val, V16);
        self.emit(Inst::FpToInt { signed: true, fp, int, rd: X9, rn: V16 });
        self.spill(X9, dest_ty)
    }
    fn uitofp(&mut self, val: Value, dest_ty: Type) -> Value {
        // Zero-extend the source to 64 bits so any integer width converts correctly.
        let i64ty = self.cx.intern_type(TypeData::Int(64));
        let wide = self.zext(val, i64ty);
        let fp = fp_size(self.cx, dest_ty);
        self.materialize(wide, X9);
        self.emit(Inst::IntToFp { signed: false, fp, int: OperandSize::S64, rd: V16, rn: X9 });
        self.spill_fp(V16, dest_ty)
    }
    fn sitofp(&mut self, val: Value, dest_ty: Type) -> Value {
        // Sign-extend the source to 64 bits so any integer width converts correctly.
        let i64ty = self.cx.intern_type(TypeData::Int(64));
        let wide = self.sext(val, i64ty);
        let fp = fp_size(self.cx, dest_ty);
        self.materialize(wide, X9);
        self.emit(Inst::IntToFp { signed: true, fp, int: OperandSize::S64, rd: V16, rn: X9 });
        self.spill_fp(V16, dest_ty)
    }
    fn fptrunc(&mut self, val: Value, dest_ty: Type) -> Value {
        let from = fp_size(self.cx, val.ty());
        let to = fp_size(self.cx, dest_ty);
        self.materialize_fp(val, V16);
        self.emit(Inst::FpCvt { from, to, rd: V16, rn: V16 });
        self.spill_fp(V16, dest_ty)
    }
    fn fpext(&mut self, val: Value, dest_ty: Type) -> Value {
        let from = fp_size(self.cx, val.ty());
        let to = fp_size(self.cx, dest_ty);
        self.materialize_fp(val, V16);
        self.emit(Inst::FpCvt { from, to, rd: V16, rn: V16 });
        self.spill_fp(V16, dest_ty)
    }
    fn ptrtoint(&mut self, val: Value, dest_ty: Type) -> Value {
        self.retype(val, dest_ty)
    }
    fn inttoptr(&mut self, val: Value, dest_ty: Type) -> Value {
        self.retype(val, dest_ty)
    }
    fn bitcast(&mut self, val: Value, dest_ty: Type) -> Value {
        self.retype(val, dest_ty)
    }
    fn intcast(&mut self, val: Value, dest_ty: Type, is_signed: bool) -> Value {
        let src_bits = match self.cx.type_data(val.ty()) {
            TypeData::Int(b) => b,
            _ => 64,
        };
        let dst_bits = match self.cx.type_data(dest_ty) {
            TypeData::Int(b) => b,
            _ => 64,
        };
        if dst_bits <= src_bits {
            self.retype(val, dest_ty)
        } else if is_signed {
            self.sext(val, dest_ty)
        } else {
            self.zext(val, dest_ty)
        }
    }
    fn pointercast(&mut self, val: Value, dest_ty: Type) -> Value {
        self.retype(val, dest_ty)
    }

    fn icmp(&mut self, op: IntPredicate, lhs: Value, rhs: Value) -> Value {
        let cond = int_pred_to_cond(op);
        self.emit_icmp(cond, lhs, rhs)
    }
    fn fcmp(&mut self, op: RealPredicate, lhs: Value, rhs: Value) -> Value {
        let size = fp_size(self.cx, lhs.ty());
        self.materialize_fp(lhs, V16);
        self.materialize_fp(rhs, V17);
        self.emit(Inst::FpCmp { size, rn: V16, rm: V17 });
        let cond = real_pred_to_cond(op);
        // cset w9, cond  ==  csinc w9, wzr, wzr, invert(cond)
        self.emit(Inst::CondSel {
            op: CondSel::Csinc,
            size: OperandSize::S32,
            rd: X9,
            rn: ZR,
            rm: ZR,
            cond: cond.invert(),
        });
        self.spill(X9, self.cx.intern_type(TypeData::Int(1)))
    }

    fn memcpy(
        &mut self,
        dst: Value,
        _dst_align: Align,
        src: Value,
        _src_align: Align,
        size: Value,
        _flags: MemFlags,
        _tt: Option<rustc_ast::expand::typetree::FncTree>,
    ) {
        self.libc_mem_call("_memcpy", dst, src, size);
    }
    fn memmove(
        &mut self,
        dst: Value,
        _dst_align: Align,
        src: Value,
        _src_align: Align,
        size: Value,
        _flags: MemFlags,
    ) {
        self.libc_mem_call("_memmove", dst, src, size);
    }
    fn memset(
        &mut self,
        ptr: Value,
        fill_byte: Value,
        size: Value,
        _align: Align,
        _flags: MemFlags,
    ) {
        // `memset(ptr, fill_byte, size)` — same argument registers as the copy helpers.
        self.libc_mem_call("_memset", ptr, fill_byte, size);
    }

    fn vscale(&mut self, _ty: Type) -> Value {
        todo!("rustc_codegen_arm64: vscale")
    }

    fn select(&mut self, cond: Value, then_val: Value, else_val: Value) -> Value {
        let ty = then_val.ty();
        let size = op_size(self.cx, ty);
        self.materialize(cond, X9);
        // cmp cond, #0 -> flags; ne means cond is true
        self.emit(Inst::AddSubImm {
            op: AddSub::Sub,
            size: OperandSize::S32,
            set_flags: true,
            rd: ZR,
            rn: X9,
            imm12: 0,
            shift12: false,
        });
        self.materialize(then_val, X10);
        self.materialize(else_val, X11_HACK);
        self.emit(Inst::CondSel {
            op: CondSel::Csel,
            size,
            rd: X9,
            rn: X10,
            rm: X11_HACK,
            cond: Cond::Ne,
        });
        self.spill(X9, ty)
    }

    fn va_arg(&mut self, _list: Value, _ty: Type) -> Value {
        todo!("rustc_codegen_arm64: va_arg")
    }
    fn extract_element(&mut self, _vec: Value, _idx: Value) -> Value {
        todo!("rustc_codegen_arm64: extract_element")
    }
    fn vector_splat(&mut self, _num_elts: usize, _elt: Value) -> Value {
        todo!("rustc_codegen_arm64: vector_splat")
    }
    fn extract_value(&mut self, agg_val: Value, idx: u64) -> Value {
        // A packed pair lives in a stack slot; field `idx` is a sub-slot at its byte offset.
        let (off, fty) = self.cx.pair_field(agg_val.ty(), idx as usize);
        match agg_val {
            Value::Slot { off: base, .. } => Value::Slot { off: base + off as u32, ty: fty },
            _ => Value::Undef { ty: fty },
        }
    }
    fn insert_value(&mut self, agg_val: Value, elt: Value, idx: u64) -> Value {
        let pair_ty = agg_val.ty();
        let (foff, _) = self.cx.pair_field(pair_ty, idx as usize);
        // Reuse the aggregate's slot if it already has one, otherwise allocate it.
        let base = match agg_val {
            Value::Slot { off, .. } => off,
            _ => {
                let (size, align) = self.cx.type_size_align(pair_ty);
                self.alloc_slot(size, align) as u32
            }
        };
        let field_off = base + foff as u32;
        if type_is_float(self.cx, elt.ty()) {
            self.materialize_fp(elt, V16);
            self.emit(Inst::LoadStoreFpUImm {
                load: false,
                size: fp_size(self.cx, elt.ty()),
                rt: V16,
                rn: SP,
                offset: field_off,
            });
        } else {
            self.materialize(elt, X9);
            self.emit(Inst::LoadStoreUImm {
                load: false,
                signed: false,
                size: mem_size(self.cx, elt.ty()),
                rt: X9,
                rn: SP,
                offset: field_off,
            });
        }
        Value::Slot { off: base, ty: pair_ty }
    }

    fn set_personality_fn(&mut self, _personality: Function) {}
    fn cleanup_landing_pad(&mut self, _pers_fn: Function) -> (Value, Value) {
        // Baseline unwinding model: cleanup landing pads abort rather than run real cleanup. The
        // landing-pad operands (exception pointer + selector) are never inspected before the
        // `resume`/terminate aborts, so undefined placeholders suffice.
        let ptr = self.ptr_ty();
        let i32_ty = self.cx.intern_type(TypeData::Int(32));
        (Value::Undef { ty: ptr }, Value::Undef { ty: i32_ty })
    }
    fn filter_landing_pad(&mut self, _pers_fn: Function) {}
    fn resume(&mut self, _exn0: Value, _exn1: Value) {
        // An unwind reaching `resume` aborts in the baseline (panic=abort semantics).
        self.emit(Inst::Brk { imm16: 1 });
    }
    fn cleanup_pad(&mut self, _parent: Option<Value>, _args: &[Value]) {}
    fn cleanup_ret(&mut self, _funclet: &(), _unwind: Option<BasicBlock>) {
        self.emit(Inst::Brk { imm16: 1 });
    }
    fn catch_pad(&mut self, _parent: Value, _args: &[Value]) {}
    fn catch_switch(
        &mut self,
        _parent: Option<Value>,
        _unwind: Option<BasicBlock>,
        _handlers: &[BasicBlock],
    ) -> Value {
        Value::Undef { ty: self.ptr_ty() }
    }
    fn get_funclet_cleanuppad(&self, _funclet: &()) -> Value {
        Value::Undef { ty: self.cx.intern_type(TypeData::Ptr) }
    }

    fn atomic_cmpxchg(
        &mut self,
        dst: Value,
        cmp: Value,
        src: Value,
        order: AtomicOrdering,
        failure_order: AtomicOrdering,
        _weak: bool,
    ) -> (Value, Value) {
        // `cas` is always strong, so it implements the weak form too.
        let ty = cmp.ty();
        let size = mem_size(self.cx, ty);
        let acquire = order_acquire(order) || order_acquire(failure_order);
        let release = order_release(order);
        let bool_ty = self.cx.intern_type(TypeData::Int(1));
        self.materialize(dst, X9); // pointer
        self.materialize(cmp, X10); // comparand (overwritten with the old value)
        self.materialize(src, X11_HACK); // new value
        self.materialize(cmp, X12); // keep the original comparand for the success test
        self.emit(Inst::AtomicCas { acquire, release, size, rs: X10, rt: X11_HACK, rn: X9 });
        // x10 now holds the old value; the swap succeeded iff it equalled the comparand.
        let cmp_size = op_size(self.cx, ty);
        self.emit(Inst::AddSubReg {
            op: AddSub::Sub,
            size: cmp_size,
            set_flags: true,
            rd: ZR,
            rn: X10,
            rm: X12,
            amount: 0,
        });
        self.emit(Inst::CondSel {
            op: CondSel::Csinc,
            size: OperandSize::S32,
            rd: X13,
            rn: ZR,
            rm: ZR,
            cond: Cond::Eq.invert(),
        });
        let old = self.spill(X10, ty);
        let success = self.spill(X13, bool_ty);
        (old, success)
    }
    fn atomic_rmw(
        &mut self,
        op: AtomicRmwBinOp,
        dst: Value,
        src: Value,
        order: AtomicOrdering,
        _ret_ptr: bool,
    ) -> Value {
        let ty = src.ty();
        let size = mem_size(self.cx, ty);
        let acquire = order_acquire(order);
        let release = order_release(order);
        // AArch64 lacks atomic subtract/and; express them as add of the negation and clear of the
        // complement, transforming the operand before the atomic.
        let (rmw_op, operand) = match op {
            AtomicRmwBinOp::AtomicXchg => (AtomicRmwOp::Swp, src),
            AtomicRmwBinOp::AtomicAdd => (AtomicRmwOp::Add, src),
            AtomicRmwBinOp::AtomicSub => (AtomicRmwOp::Add, self.neg(src)),
            AtomicRmwBinOp::AtomicAnd => (AtomicRmwOp::Clr, self.not(src)),
            AtomicRmwBinOp::AtomicOr => (AtomicRmwOp::Set, src),
            AtomicRmwBinOp::AtomicXor => (AtomicRmwOp::Eor, src),
            AtomicRmwBinOp::AtomicMax => (AtomicRmwOp::Smax, src),
            AtomicRmwBinOp::AtomicMin => (AtomicRmwOp::Smin, src),
            AtomicRmwBinOp::AtomicUMax => (AtomicRmwOp::Umax, src),
            AtomicRmwBinOp::AtomicUMin => (AtomicRmwOp::Umin, src),
            // `nand` has no single LSE op (it needs a load/store-exclusive loop); deferred.
            AtomicRmwBinOp::AtomicNand => todo!("rustc_codegen_arm64: atomic nand"),
        };
        self.materialize(dst, X9); // pointer
        self.materialize(operand, X10); // rs
        self.emit(Inst::AtomicRmw {
            op: rmw_op,
            acquire,
            release,
            size,
            rs: X10,
            rt: X11_HACK,
            rn: X9,
        });
        self.spill(X11_HACK, ty)
    }
    fn atomic_fence(&mut self, order: AtomicOrdering, _scope: SynchronizationScope) {
        let option = match order {
            AtomicOrdering::Acquire => DmbOption::IshLd,
            _ => DmbOption::Ish,
        };
        self.emit(Inst::Dmb { option });
    }
    fn set_invariant_load(&mut self, _load: Value) {}

    fn lifetime_start(&mut self, _ptr: Value, _size: Size) {}
    fn lifetime_end(&mut self, _ptr: Value, _size: Size) {}

    fn call(
        &mut self,
        _llty: Type,
        _caller_attrs: Option<&CodegenFnAttrs>,
        fn_abi: Option<&FnAbi<'tcx, Ty<'tcx>>>,
        fn_val: Value,
        args: &[Value],
        _funclet: Option<&()>,
        _callee_instance: Option<Instance<'tcx>>,
    ) -> Value {
        self.emit_call_core(fn_abi, fn_val, args)
    }
    fn tail_call(
        &mut self,
        _llty: Type,
        _caller_attrs: Option<&CodegenFnAttrs>,
        _fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        _llfn: Value,
        _args: &[Value],
        _funclet: Option<&()>,
        _callee_instance: Option<Instance<'tcx>>,
    ) {
        todo!("rustc_codegen_arm64: tail_call")
    }

    fn apply_attrs_to_cleanup_callsite(&mut self, _llret: Value) {}
}

/// Scratch register used by a few helpers that need a third temporary.
const X11_HACK: Gpr = Gpr::from_encoding(11);

impl<'a, 'tcx> Builder<'a, 'tcx> {
    fn rem(&mut self, lhs: Value, rhs: Value, div: DataProc2) -> Value {
        let ty = lhs.ty();
        let size = op_size(self.cx, ty);
        self.materialize(lhs, X9);
        self.materialize(rhs, X10);
        // q = lhs / rhs ; rem = lhs - q*rhs  ==  msub rem, q, rhs, lhs
        self.emit(Inst::DataProc2 { op: div, size, rd: X11_HACK, rn: X9, rm: X10 });
        self.emit(Inst::Msub { size, rd: X9, rn: X11_HACK, rm: X10, ra: X9 });
        self.spill(X9, ty)
    }

    /// Emit a call to a libc memory helper with the C signature `(ptr x0, arg x1, size x2)`
    /// (covers `memcpy`/`memmove`/`memset`). The arguments are loaded into `x0..x2` and the call
    /// goes through a `bl` to an undefined external symbol the linker resolves against libc.
    fn libc_mem_call(&mut self, sym: &str, a0: Value, a1: Value, a2: Value) {
        self.materialize(a0, X0);
        self.materialize(a1, X1);
        self.materialize(a2, X2);
        self.emit(Inst::Bl { sym: SymRef::new(sym) });
    }

    /// Load a scalar pair from memory at `ptr` into a fresh packed slot, field by field.
    fn load_pair(&mut self, ty: Type, ptr: Value) -> Value {
        let (size, align) = self.cx.type_size_align(ty);
        let base = self.alloc_slot(size, align) as u32;
        self.materialize(ptr, X9);
        for idx in 0..2 {
            let (foff, fty) = self.cx.pair_field(ty, idx);
            let foff = foff as u32;
            if type_is_float(self.cx, fty) {
                let sz = fp_size(self.cx, fty);
                self.emit(Inst::LoadStoreFpUImm { load: true, size: sz, rt: V16, rn: X9, offset: foff });
                self.emit(Inst::LoadStoreFpUImm { load: false, size: sz, rt: V16, rn: SP, offset: base + foff });
            } else {
                let sz = mem_size(self.cx, fty);
                self.emit(Inst::LoadStoreUImm { load: true, signed: false, size: sz, rt: X10, rn: X9, offset: foff });
                self.emit(Inst::LoadStoreUImm { load: false, signed: false, size: sz, rt: X10, rn: SP, offset: base + foff });
            }
        }
        Value::Slot { off: base, ty }
    }

    /// Store a packed scalar pair `val` (a slot) to memory at `ptr`, field by field.
    fn store_pair(&mut self, val: Value, ptr: Value) {
        let ty = val.ty();
        let Value::Slot { off: sbase, .. } = val else { return };
        self.materialize(ptr, X9);
        for idx in 0..2 {
            let (foff, fty) = self.cx.pair_field(ty, idx);
            let foff = foff as u32;
            if type_is_float(self.cx, fty) {
                let sz = fp_size(self.cx, fty);
                self.emit(Inst::LoadStoreFpUImm { load: true, size: sz, rt: V16, rn: SP, offset: sbase + foff });
                self.emit(Inst::LoadStoreFpUImm { load: false, size: sz, rt: V16, rn: X9, offset: foff });
            } else {
                let sz = mem_size(self.cx, fty);
                self.emit(Inst::LoadStoreUImm { load: true, signed: false, size: sz, rt: X10, rn: SP, offset: sbase + foff });
                self.emit(Inst::LoadStoreUImm { load: false, signed: false, size: sz, rt: X10, rn: X9, offset: foff });
            }
        }
    }

    /// Shared implementation of `call`/`invoke`: marshal arguments into the ABI registers, emit the
    /// branch-and-link (direct or indirect), and collect the return value.
    fn emit_call_core(
        &mut self,
        fn_abi: Option<&FnAbi<'tcx, Ty<'tcx>>>,
        fn_val: Value,
        args: &[Value],
    ) -> Value {
        let indirect_ret = fn_abi.is_some_and(|a| a.ret.is_indirect());
        // Marshal physical arguments into the integer (`x0..x7`) and floating-point (`v0..v7`)
        // register banks; an sret pointer goes in x8.
        let mut ngrn: u8 = 0;
        let mut nsrn: u8 = 0;
        for (i, &arg) in args.iter().enumerate() {
            if indirect_ret && i == 0 {
                self.materialize(arg, Gpr::from_encoding(8));
            } else if type_is_float(self.cx, arg.ty()) {
                self.materialize_fp(arg, Vreg::from_encoding(nsrn));
                nsrn += 1;
            } else {
                self.materialize(arg, Gpr::from_encoding(ngrn));
                ngrn += 1;
            }
        }
        // Emit the call. Direct calls reference the symbol (BRANCH26 relocation); otherwise the
        // callee address is materialized and called indirectly.
        match fn_val {
            Value::Sym { sym, offset, .. } => {
                let name = self.cx.sym_name(sym);
                self.emit(Inst::Bl { sym: SymRef { name, addend: offset } });
            }
            other => {
                self.materialize(other, X9);
                self.emit(Inst::Blr { rn: X9 });
            }
        }
        // Collect the return value from x0 / v0 (scalar baseline).
        match fn_abi {
            Some(a) if !a.ret.is_indirect() && !a.ret.is_ignore() => {
                let ty = self.cx.immediate_backend_type(a.ret.layout);
                match self.cx.type_data(ty) {
                    TypeData::Pair(fa, fb) => {
                        // `PassMode::Pair`: field 0 comes back in x0/v0, field 1 in x1/v1; pack
                        // them into a fresh slot.
                        let (size, align) = self.cx.type_size_align(ty);
                        let base = self.alloc_slot(size, align) as u32;
                        let mut ngrn: u8 = 0;
                        let mut nsrn: u8 = 0;
                        for (idx, fty) in [(0usize, fa), (1usize, fb)] {
                            let (foff, _) = self.cx.pair_field(ty, idx);
                            let field_off = base + foff as u32;
                            if type_is_float(self.cx, fty) {
                                self.emit(Inst::LoadStoreFpUImm {
                                    load: false,
                                    size: fp_size(self.cx, fty),
                                    rt: Vreg::from_encoding(nsrn),
                                    rn: SP,
                                    offset: field_off,
                                });
                                nsrn += 1;
                            } else {
                                self.emit(Inst::LoadStoreUImm {
                                    load: false,
                                    signed: false,
                                    size: mem_size(self.cx, fty),
                                    rt: Gpr::from_encoding(ngrn),
                                    rn: SP,
                                    offset: field_off,
                                });
                                ngrn += 1;
                            }
                        }
                        Value::Slot { off: base, ty }
                    }
                    _ if type_is_float(self.cx, ty) => self.spill_fp(V0, ty),
                    _ => self.spill(X0, ty),
                }
            }
            _ => Value::Undef { ty: self.ptr_ty() },
        }
    }

    /// Multiplication with overflow detection, returning `(low_product, overflowed)`.
    ///
    /// For 64-bit operands the high half of the 128-bit product is computed with `umulh`/`smulh`
    /// and compared against the sign-extension of the low half (signed) or zero (unsigned). For
    /// narrower widths the operands are extended to 64 bits, multiplied once, and the result is
    /// range-checked against the type width.
    fn checked_mul(
        &mut self,
        signed: bool,
        val_ty: Type,
        lhs: Value,
        rhs: Value,
    ) -> (Value, Value) {
        let bool_ty = self.cx.intern_type(TypeData::Int(1));
        let width = match self.cx.type_data(val_ty) {
            TypeData::Int(b) => b,
            _ => 64,
        };
        let overflow_cond = if width == 64 {
            self.materialize(lhs, X9);
            self.materialize(rhs, X10);
            // high = mulhi(a, b); low = a * b (the result).
            self.emit(Inst::MulHigh { signed, rd: X12, rn: X9, rm: X10 });
            self.emit(Inst::Madd { size: OperandSize::S64, rd: X9, rn: X9, rm: X10, ra: ZR });
            if signed {
                // Overflow iff high != (low >>s 63).
                self.load_imm(X13, 63, OperandSize::S64);
                self.emit(Inst::DataProc2 {
                    op: DataProc2::Asrv,
                    size: OperandSize::S64,
                    rd: X13,
                    rn: X9,
                    rm: X13,
                });
                self.emit(Inst::AddSubReg {
                    op: AddSub::Sub,
                    size: OperandSize::S64,
                    set_flags: true,
                    rd: ZR,
                    rn: X12,
                    rm: X13,
                    amount: 0,
                });
            } else {
                // Overflow iff high != 0.
                self.emit(Inst::AddSubReg {
                    op: AddSub::Sub,
                    size: OperandSize::S64,
                    set_flags: true,
                    rd: ZR,
                    rn: X12,
                    rm: ZR,
                    amount: 0,
                });
            }
            Cond::Ne
        } else {
            // Widen to 64 bits so the full product fits, then range-check.
            let i64ty = self.cx.intern_type(TypeData::Int(64));
            let a = if signed { self.sext(lhs, i64ty) } else { self.zext(lhs, i64ty) };
            let b = if signed { self.sext(rhs, i64ty) } else { self.zext(rhs, i64ty) };
            self.materialize(a, X9);
            self.materialize(b, X10);
            self.emit(Inst::Madd { size: OperandSize::S64, rd: X9, rn: X9, rm: X10, ra: ZR });
            if signed {
                // Overflow iff the product differs from the sign-extension of its low `width` bits.
                let shift = (64 - width) as u128;
                self.load_imm(X13, shift, OperandSize::S64);
                self.emit(Inst::DataProc2 {
                    op: DataProc2::Lslv,
                    size: OperandSize::S64,
                    rd: X12,
                    rn: X9,
                    rm: X13,
                });
                self.emit(Inst::DataProc2 {
                    op: DataProc2::Asrv,
                    size: OperandSize::S64,
                    rd: X12,
                    rn: X12,
                    rm: X13,
                });
                self.emit(Inst::AddSubReg {
                    op: AddSub::Sub,
                    size: OperandSize::S64,
                    set_flags: true,
                    rd: ZR,
                    rn: X9,
                    rm: X12,
                    amount: 0,
                });
            } else {
                // Overflow iff any bit above `width` is set.
                self.load_imm(X13, width as u128, OperandSize::S64);
                self.emit(Inst::DataProc2 {
                    op: DataProc2::Lsrv,
                    size: OperandSize::S64,
                    rd: X12,
                    rn: X9,
                    rm: X13,
                });
                self.emit(Inst::AddSubReg {
                    op: AddSub::Sub,
                    size: OperandSize::S64,
                    set_flags: true,
                    rd: ZR,
                    rn: X12,
                    rm: ZR,
                    amount: 0,
                });
            }
            Cond::Ne
        };
        // cset overflow, cond  ==  csinc overflow, wzr, wzr, invert(cond)
        self.emit(Inst::CondSel {
            op: CondSel::Csinc,
            size: OperandSize::S32,
            rd: X11_HACK,
            rn: ZR,
            rm: ZR,
            cond: overflow_cond.invert(),
        });
        let result = self.spill(X9, val_ty);
        let overflow = self.spill(X11_HACK, bool_ty);
        (result, overflow)
    }
}
