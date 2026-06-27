//! The `Builder`: per-function machine-code construction.
//!
//! The baseline model keeps every SSA value in a stack slot (no register allocator). A `Builder`
//! appends [`Inst`]s to the current basic block of the [`FunctionBuild`] held by the context; when
//! the function is finished its blocks are concatenated (each prefixed with its label) into a
//! [`MachFunction`] and pushed to the module.

use std::ops::Deref;

use rustc_abi::{Align, HasDataLayout, Scalar, Size, TargetDataLayout, WrappingRange};
use rustc_ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_codegen_ssa::common::{
    AtomicRmwBinOp, IntPredicate, RealPredicate, SynchronizationScope,
};
use rustc_codegen_ssa::mir::IntrinsicResult;
use rustc_codegen_ssa::mir::operand::{OperandRef, OperandValue};
use rustc_codegen_ssa::mir::place::{PlaceRef, PlaceValue};
use rustc_codegen_ssa::traits::{
    AbiBuilderMethods, ArgAbiBuilderMethods, AsmBuilderMethods, BackendTypes, BuilderMethods,
    CoverageInfoBuilderMethods, DebugInfoBuilderMethods, InlineAsmOperandRef,
    IntrinsicCallBuilderMethods, LayoutTypeCodegenMethods, OverflowOp, StaticBuilderMethods,
};
use rustc_codegen_ssa::{MemFlags, RetagInfo};
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs;
use rustc_middle::mir::coverage::CoverageKind;
use rustc_middle::ty::layout::{
    FnAbiOf, FnAbiOfHelpers, HasTyCtxt, HasTypingEnv, LayoutOfHelpers, TyAndLayout,
};
use rustc_middle::ty::{self, AtomicOrdering, Instance, Ty, TyCtxt};
use rustc_span::Span;
use rustc_target::callconv::{ArgAbi, FnAbi, PassMode};
use rustc_target::spec::{HasTargetSpec, Target};

use crate::context::{BasicBlock, CodegenCx, Function, Type, TypeData, Value};
use crate::mach::func::MachFunction;
use crate::mach::inst::{
    AddSub, CondSel, DataProc2, Inst, Label, LogicOp, MemSize, MovKind, PairIndex, SymRef,
};
use crate::mach::reg::{Cond, Gpr, OperandSize, FP, LR, SP, X0, X9, X10, ZR};

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

/// Compute the incoming register and backend type of each physical parameter (baseline: integer
/// args in `x0..x7`, indirect return pointer in `x8`). Floating-point and stack-passed arguments
/// are not yet handled.
fn build_param_list<'tcx>(
    cx: &CodegenCx<'tcx>,
    fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
) -> Vec<(Gpr, Type)> {
    let mut params = Vec::new();
    let ptr = cx.intern_type(TypeData::Ptr);
    if fn_abi.ret.is_indirect() {
        // The indirect-return (sret) pointer arrives in x8.
        params.push((Gpr::from_encoding(8), ptr));
    }
    let mut ngrn: u8 = 0;
    for arg in fn_abi.args.iter() {
        match arg.mode {
            PassMode::Ignore => {}
            PassMode::Direct(_) => {
                let ty = cx.immediate_backend_type(arg.layout);
                params.push((Gpr::from_encoding(ngrn), ty));
                ngrn += 1;
            }
            PassMode::Pair(..) => {
                let a = cx.scalar_pair_element_backend_type(arg.layout, 0, true);
                let b = cx.scalar_pair_element_backend_type(arg.layout, 1, true);
                params.push((Gpr::from_encoding(ngrn), a));
                ngrn += 1;
                params.push((Gpr::from_encoding(ngrn), b));
                ngrn += 1;
            }
            PassMode::Indirect { .. } => {
                params.push((Gpr::from_encoding(ngrn), ptr));
                ngrn += 1;
            }
            PassMode::Cast { .. } => {
                params.push((Gpr::from_encoding(ngrn), cx.intern_type(TypeData::Int(64))));
                ngrn += 1;
            }
        }
    }
    params
}

/// Spill each incoming parameter register into a frame slot at function entry and record the slots.
fn setup_params(cx: &CodegenCx<'_>, fb: &mut FunctionBuild, block: BasicBlock) {
    let Some(instance) = cx.cur_instance.get() else { return };
    let fn_abi = cx.fn_abi_of_instance(instance, ty::List::empty());
    for (reg, ty) in build_param_list(cx, fn_abi) {
        let (size, align) = cx.type_size_align(ty);
        let off = fb.frame.alloc(size, align) as u32;
        fb.blocks[block.0 as usize].push(Inst::LoadStoreUImm {
            load: false,
            signed: false,
            size: mem_size(cx, ty),
            rt: reg,
            rn: SP,
            offset: off,
        });
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
        let bits = bits as u64;
        self.emit(Inst::MovWide {
            kind: MovKind::Zero,
            size,
            rd: reg,
            imm16: (bits & 0xffff) as u16,
            shift: 0,
        });
        for shift in [16u8, 32, 48] {
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
        _args: &[OperandRef<'tcx, Value>],
        _result_layout: TyAndLayout<'tcx>,
        _result_place: Option<PlaceValue<Value>>,
        _span: Span,
    ) -> IntrinsicResult<'tcx, Value> {
        // Fall back to the intrinsic's MIR body for now.
        IntrinsicResult::Fallback(instance)
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
    fn get_static(&mut self, _def_id: rustc_hir::def_id::DefId) -> Value {
        todo!("rustc_codegen_arm64: get_static")
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
        self.materialize(v, X0);
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
        _fn_abi: Option<&FnAbi<'tcx, Ty<'tcx>>>,
        _llfn: Value,
        _args: &[Value],
        _then: BasicBlock,
        _catch: BasicBlock,
        _funclet: Option<&()>,
        _instance: Option<Instance<'tcx>>,
    ) -> Value {
        todo!("rustc_codegen_arm64: invoke")
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
    fn fneg(&mut self, _v: Value) -> Value {
        todo!("rustc_codegen_arm64: fneg")
    }
    fn not(&mut self, v: Value) -> Value {
        let all_ones = Value::Const { bits: u128::MAX, ty: v.ty() };
        self.xor(v, all_ones)
    }

    fn fadd(&mut self, _lhs: Value, _rhs: Value) -> Value {
        todo!("rustc_codegen_arm64: fadd")
    }
    fn fadd_fast(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fadd(lhs, rhs)
    }
    fn fadd_algebraic(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fadd(lhs, rhs)
    }
    fn fsub(&mut self, _lhs: Value, _rhs: Value) -> Value {
        todo!("rustc_codegen_arm64: fsub")
    }
    fn fsub_fast(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fsub(lhs, rhs)
    }
    fn fsub_algebraic(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fsub(lhs, rhs)
    }
    fn fmul(&mut self, _lhs: Value, _rhs: Value) -> Value {
        todo!("rustc_codegen_arm64: fmul")
    }
    fn fmul_fast(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fmul(lhs, rhs)
    }
    fn fmul_algebraic(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fmul(lhs, rhs)
    }
    fn fdiv(&mut self, _lhs: Value, _rhs: Value) -> Value {
        todo!("rustc_codegen_arm64: fdiv")
    }
    fn fdiv_fast(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fdiv(lhs, rhs)
    }
    fn fdiv_algebraic(&mut self, lhs: Value, rhs: Value) -> Value {
        self.fdiv(lhs, rhs)
    }
    fn frem(&mut self, _lhs: Value, _rhs: Value) -> Value {
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
        let size = op_size(self.cx, lhs.ty());
        let val_ty = lhs.ty();
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
            OverflowOp::Mul => {
                // FIXME: detect multiplication overflow via `umulh`/`smulh`. For now compute the
                // low product and report no overflow.
                self.emit(Inst::Madd { size, rd: X9, rn: X9, rm: X10, ra: ZR });
                let result = self.spill(X9, val_ty);
                let overflow = Value::Const { bits: 0, ty: bool_ty };
                return (result, overflow);
            }
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
    fn atomic_load(&mut self, _ty: Type, _ptr: Value, _order: AtomicOrdering, _size: Size) -> Value {
        todo!("rustc_codegen_arm64: atomic_load")
    }
    fn load_operand(&mut self, place: PlaceRef<'tcx, Value>) -> OperandRef<'tcx, Value> {
        if place.layout.is_zst() {
            return OperandRef::zero_sized(place.layout);
        }
        if self.cx.is_backend_immediate(place.layout) {
            let ty = self.cx.immediate_backend_type(place.layout);
            let val = self.load(ty, place.val.llval, place.val.align);
            OperandRef {
                val: OperandValue::Immediate(val),
                layout: place.layout,
                move_annotation: None,
            }
        } else {
            OperandRef {
                val: OperandValue::Ref(place.val),
                layout: place.layout,
                move_annotation: None,
            }
        }
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
    fn atomic_store(&mut self, _val: Value, _ptr: Value, _order: AtomicOrdering, _size: Size) {
        todo!("rustc_codegen_arm64: atomic_store")
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
    fn fptoui_sat(&mut self, _val: Value, _dest_ty: Type) -> Value {
        todo!("rustc_codegen_arm64: fptoui_sat")
    }
    fn fptosi_sat(&mut self, _val: Value, _dest_ty: Type) -> Value {
        todo!("rustc_codegen_arm64: fptosi_sat")
    }
    fn fptoui(&mut self, _val: Value, _dest_ty: Type) -> Value {
        todo!("rustc_codegen_arm64: fptoui")
    }
    fn fptosi(&mut self, _val: Value, _dest_ty: Type) -> Value {
        todo!("rustc_codegen_arm64: fptosi")
    }
    fn uitofp(&mut self, _val: Value, _dest_ty: Type) -> Value {
        todo!("rustc_codegen_arm64: uitofp")
    }
    fn sitofp(&mut self, _val: Value, _dest_ty: Type) -> Value {
        todo!("rustc_codegen_arm64: sitofp")
    }
    fn fptrunc(&mut self, _val: Value, _dest_ty: Type) -> Value {
        todo!("rustc_codegen_arm64: fptrunc")
    }
    fn fpext(&mut self, _val: Value, _dest_ty: Type) -> Value {
        todo!("rustc_codegen_arm64: fpext")
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
    fn fcmp(&mut self, _op: RealPredicate, _lhs: Value, _rhs: Value) -> Value {
        todo!("rustc_codegen_arm64: fcmp")
    }

    fn memcpy(
        &mut self,
        _dst: Value,
        _dst_align: Align,
        _src: Value,
        _src_align: Align,
        _size: Value,
        _flags: MemFlags,
        _tt: Option<rustc_ast::expand::typetree::FncTree>,
    ) {
        todo!("rustc_codegen_arm64: memcpy")
    }
    fn memmove(
        &mut self,
        _dst: Value,
        _dst_align: Align,
        _src: Value,
        _src_align: Align,
        _size: Value,
        _flags: MemFlags,
    ) {
        todo!("rustc_codegen_arm64: memmove")
    }
    fn memset(
        &mut self,
        _ptr: Value,
        _fill_byte: Value,
        _size: Value,
        _align: Align,
        _flags: MemFlags,
    ) {
        todo!("rustc_codegen_arm64: memset")
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
    fn extract_value(&mut self, _agg_val: Value, _idx: u64) -> Value {
        todo!("rustc_codegen_arm64: extract_value")
    }
    fn insert_value(&mut self, _agg_val: Value, _elt: Value, _idx: u64) -> Value {
        todo!("rustc_codegen_arm64: insert_value")
    }

    fn set_personality_fn(&mut self, _personality: Function) {}
    fn cleanup_landing_pad(&mut self, _pers_fn: Function) -> (Value, Value) {
        todo!("rustc_codegen_arm64: cleanup_landing_pad")
    }
    fn filter_landing_pad(&mut self, _pers_fn: Function) {
        todo!("rustc_codegen_arm64: filter_landing_pad")
    }
    fn resume(&mut self, _exn0: Value, _exn1: Value) {
        todo!("rustc_codegen_arm64: resume")
    }
    fn cleanup_pad(&mut self, _parent: Option<Value>, _args: &[Value]) {}
    fn cleanup_ret(&mut self, _funclet: &(), _unwind: Option<BasicBlock>) {
        todo!("rustc_codegen_arm64: cleanup_ret")
    }
    fn catch_pad(&mut self, _parent: Value, _args: &[Value]) {}
    fn catch_switch(
        &mut self,
        _parent: Option<Value>,
        _unwind: Option<BasicBlock>,
        _handlers: &[BasicBlock],
    ) -> Value {
        todo!("rustc_codegen_arm64: catch_switch")
    }
    fn get_funclet_cleanuppad(&self, _funclet: &()) -> Value {
        todo!("rustc_codegen_arm64: get_funclet_cleanuppad")
    }

    fn atomic_cmpxchg(
        &mut self,
        _dst: Value,
        _cmp: Value,
        _src: Value,
        _order: AtomicOrdering,
        _failure_order: AtomicOrdering,
        _weak: bool,
    ) -> (Value, Value) {
        todo!("rustc_codegen_arm64: atomic_cmpxchg")
    }
    fn atomic_rmw(
        &mut self,
        _op: AtomicRmwBinOp,
        _dst: Value,
        _src: Value,
        _order: AtomicOrdering,
        _ret_ptr: bool,
    ) -> Value {
        todo!("rustc_codegen_arm64: atomic_rmw")
    }
    fn atomic_fence(&mut self, _order: AtomicOrdering, _scope: SynchronizationScope) {
        todo!("rustc_codegen_arm64: atomic_fence")
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
        let indirect_ret = fn_abi.is_some_and(|a| a.ret.is_indirect());
        // Marshal physical arguments into x0..x7 (integer baseline); sret pointer goes in x8.
        let mut ngrn: u8 = 0;
        for (i, &arg) in args.iter().enumerate() {
            let reg = if indirect_ret && i == 0 {
                Gpr::from_encoding(8)
            } else {
                let r = Gpr::from_encoding(ngrn);
                ngrn += 1;
                r
            };
            self.materialize(arg, reg);
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
        // Collect the return value from x0 (integer/pointer baseline).
        match fn_abi {
            Some(a) if !a.ret.is_indirect() && !a.ret.is_ignore() => {
                let ty = self.cx.immediate_backend_type(a.ret.layout);
                self.spill(X0, ty)
            }
            _ => Value::Undef { ty: self.ptr_ty() },
        }
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
}
