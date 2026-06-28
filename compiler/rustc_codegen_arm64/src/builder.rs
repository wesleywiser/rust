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
use rustc_hir::def::DefKind;
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
use crate::mach::frame::FrameLayout;
use crate::mach::inst::{
    AddSub, AtomicRmwOp, CondSel, DataProc1, DataProc2, DmbOption, FpOp1, FpOp2, Inst, Label,
    LogicOp, MemSize, MovKind, PairIndex, SymRef,
};
use crate::mach::reg::{
    Cond, FpSize, Gpr, OperandSize, Vreg, FP, LR, SP, V0, V1, V16, V17, X0, X1, X2, X3, X9, X10,
    X11, X12, X13, X16, ZR,
};

/// State for the function currently being lowered.
pub struct FunctionBuild {
    pub func: Function,
    pub name: Box<str>,
    pub is_global: bool,
    /// Instruction list for each basic block, indexed by block id (which is also its [`Label`]).
    pub blocks: Vec<Vec<Inst>>,
    pub frame: FrameLayout,
    /// Spilled location of each physical incoming parameter, indexed by physical param index.
    pub param_slots: Vec<Value>,
}

impl FunctionBuild {
    /// Create a function builder. `outgoing_bytes` is the size of the outgoing-argument area
    /// (computed up front by scanning the function's calls); it fixes where local slots begin.
    pub fn new(
        func: Function,
        name: Box<str>,
        is_global: bool,
        outgoing_bytes: u64,
    ) -> FunctionBuild {
        FunctionBuild {
            func,
            name,
            is_global,
            blocks: Vec::new(),
            frame: FrameLayout::new(outgoing_bytes),
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
        push_sp_adjust(&mut f.insts, AddSub::Sub, frame);

        for (id, block) in self.blocks.into_iter().enumerate() {
            f.push(Inst::Label(id as Label));
            for inst in block {
                if let Inst::Ret { .. } = inst {
                    // Epilogue before every return.
                    push_sp_adjust(&mut f.insts, AddSub::Add, frame);
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

/// Whether a byte `offset` fits the scaled 12-bit unsigned-immediate load/store form for an access
/// of `size_bytes` bytes (the encoded field holds `offset / size_bytes`, which must be < 4096).
fn imm_offset_fits(offset: u64, size_bytes: u64) -> bool {
    offset % size_bytes == 0 && offset / size_bytes < (1 << 12)
}

/// Push a `movz`/`movk` sequence materializing the 64-bit `value` into `reg`.
fn push_load_imm64(out: &mut Vec<Inst>, reg: Gpr, value: u64) {
    out.push(Inst::MovWide {
        kind: MovKind::Zero,
        size: OperandSize::S64,
        rd: reg,
        imm16: (value & 0xffff) as u16,
        shift: 0,
    });
    for shift in [16u8, 32, 48] {
        let chunk = ((value >> shift) & 0xffff) as u16;
        if chunk != 0 {
            out.push(Inst::MovWide {
                kind: MovKind::Keep,
                size: OperandSize::S64,
                rd: reg,
                imm16: chunk,
                shift,
            });
        }
    }
}

/// Push a GPR load/store of `rt` at `[base, #offset]`. Offsets too large for the scaled-immediate
/// form have their address formed in the `x16` scratch first (`base + offset`, extended-register
/// add so `sp` is allowed), then the access is done at `[x16]`.
fn push_mem_gpr(
    out: &mut Vec<Inst>,
    load: bool,
    signed: bool,
    size: MemSize,
    rt: Gpr,
    base: Gpr,
    offset: u64,
) {
    if imm_offset_fits(offset, size.bytes() as u64) {
        out.push(Inst::LoadStoreUImm { load, signed, size, rt, rn: base, offset });
    } else {
        push_load_imm64(out, X16, offset);
        out.push(Inst::AddSubExtReg { op: AddSub::Add, size: OperandSize::S64, rd: X16, rn: base, rm: X16 });
        out.push(Inst::LoadStoreUImm { load, signed, size, rt, rn: X16, offset: 0 });
    }
}

/// Push a floating-point load/store of `rt` at `[base, #offset]`, with the same large-offset
/// handling as [`push_mem_gpr`].
fn push_mem_fp(out: &mut Vec<Inst>, load: bool, size: FpSize, rt: Vreg, base: Gpr, offset: u64) {
    let bytes = match size {
        FpSize::S32 => 4,
        FpSize::S64 => 8,
    };
    if imm_offset_fits(offset, bytes) {
        out.push(Inst::LoadStoreFpUImm { load, size, rt, rn: base, offset });
    } else {
        push_load_imm64(out, X16, offset);
        out.push(Inst::AddSubExtReg { op: AddSub::Add, size: OperandSize::S64, rd: X16, rn: base, rm: X16 });
        out.push(Inst::LoadStoreFpUImm { load, size, rt, rn: X16, offset: 0 });
    }
}

/// Push the stack-pointer adjustment for the prologue (`Sub`) or epilogue (`Add`). A frame too large
/// for a single 12-bit immediate materializes the size into `x16` and uses the extended-register
/// form (which, unlike the shifted form, permits `sp` as the destination/source).
fn push_sp_adjust(out: &mut Vec<Inst>, op: AddSub, frame: u64) {
    if frame == 0 {
        return;
    }
    if frame < (1 << 12) {
        out.push(Inst::AddSubImm {
            op,
            size: OperandSize::S64,
            set_flags: false,
            rd: SP,
            rn: SP,
            imm12: frame as u16,
            shift12: false,
        });
    } else {
        push_load_imm64(out, X16, frame);
        out.push(Inst::AddSubExtReg { op, size: OperandSize::S64, rd: SP, rn: SP, rm: X16 });
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

    /// Emit a GPR load/store of `rt` at `[base, #offset]`, handling offsets too large for the
    /// scaled-immediate form (see [`push_mem_gpr`]).
    fn emit_mem_gpr(&mut self, load: bool, signed: bool, size: MemSize, rt: Gpr, base: Gpr, offset: u64) {
        let mut cur = self.cx.cur_fn.borrow_mut();
        let fb = cur.as_mut().expect("no function is currently being built");
        push_mem_gpr(&mut fb.blocks[self.block.0 as usize], load, signed, size, rt, base, offset);
    }

    /// Emit a floating-point load/store of `rt` at `[base, #offset]` (see [`push_mem_fp`]).
    fn emit_mem_fp(&mut self, load: bool, size: FpSize, rt: Vreg, base: Gpr, offset: u64) {
        let mut cur = self.cx.cur_fn.borrow_mut();
        let fb = cur.as_mut().expect("no function is currently being built");
        push_mem_fp(&mut fb.blocks[self.block.0 as usize], load, size, rt, base, offset);
    }

    /// Form the address of a frame slot (`sp + off`) into `rd`. Offsets too large for the add's
    /// 12-bit immediate are materialized into `rd` first, then added with the extended-register
    /// form (which, unlike a shifted register, permits `sp` as the source).
    fn emit_frame_addr(&mut self, rd: Gpr, off: u64) {
        if off < (1 << 12) {
            self.emit(Inst::AddSubImm {
                op: AddSub::Add,
                size: OperandSize::S64,
                set_flags: false,
                rd,
                rn: SP,
                imm12: off as u16,
                shift12: false,
            });
        } else {
            self.load_imm(rd, off as u128, OperandSize::S64);
            self.emit(Inst::AddSubExtReg { op: AddSub::Add, size: OperandSize::S64, rd, rn: SP, rm: rd });
        }
    }

    /// Reserve a stack slot of the given size/alignment, returning its `sp`-relative offset.
    pub fn alloc_slot(&self, size: u64, align: u64) -> u64 {
        let mut cur = self.cx.cur_fn.borrow_mut();
        let fb = cur.as_mut().expect("no function is currently being built");
        fb.frame.alloc_local(size, align)
    }

    /// Copy `size` bytes from `[src]` into the frame slot at `dst_off`, using descending
    /// power-of-two chunks through the `x10` scratch (`src` must not be `x10`/`x16`).
    fn copy_ptr_to_slot(&mut self, dst_off: u64, src: Gpr, size: u64) {
        let mut o = 0u64;
        for chunk in [8u64, 4, 2, 1] {
            let msize = mem_size_from_bytes(chunk);
            while o + chunk <= size {
                self.emit(Inst::LoadStoreUImm {
                    load: true,
                    signed: false,
                    size: msize,
                    rt: X10,
                    rn: src,
                    offset: o,
                });
                self.emit_mem_gpr(false, false, msize, X10, SP, dst_off + o);
                o += chunk;
            }
        }
    }

    /// Copy `size` bytes from the frame slot at `src_off` into `[dst]` (mirror of
    /// [`copy_ptr_to_slot`](Self::copy_ptr_to_slot)).
    fn copy_slot_to_ptr(&mut self, src_off: u64, dst: Gpr, size: u64) {
        let mut o = 0u64;
        for chunk in [8u64, 4, 2, 1] {
            let msize = mem_size_from_bytes(chunk);
            while o + chunk <= size {
                self.emit_mem_gpr(true, false, msize, X10, SP, src_off + o);
                self.emit(Inst::LoadStoreUImm {
                    load: false,
                    signed: false,
                    size: msize,
                    rt: X10,
                    rn: dst,
                    offset: o,
                });
                o += chunk;
            }
        }
    }

    /// The lane element type, lane count, and element byte size of a vector backend type.
    fn vector_info(&self, vec_ty: Type) -> (Type, u64, u64) {
        match self.cx.type_data(vec_ty) {
            TypeData::Vector(elem, count) => {
                let (es, _) = self.cx.type_size_align(elem);
                (elem, count, es)
            }
            other => panic!("rustc_codegen_arm64: expected a vector type, found {other:?}"),
        }
    }

    /// The `sp`-relative offset of a vector operand's data. Runtime vectors already live in a frame
    /// slot; a constant vector (a `Sym` pointing at read-only data, produced by `const_vector`) is
    /// copied into a fresh slot first.
    fn vector_to_slot(&mut self, val: Value) -> Option<u64> {
        match val {
            Value::Slot { off, .. } => Some(off),
            Value::Sym { sym, offset, ty } => {
                let (size, align) = self.cx.type_size_align(ty);
                let off = self.alloc_slot(size, align);
                let symref = SymRef { name: self.cx.sym_name(sym), addend: offset };
                self.emit(Inst::Adrp { rd: X9, sym: symref.clone() });
                self.emit(Inst::AddLo { rd: X9, rn: X9, sym: symref });
                self.copy_ptr_to_slot(off, X9, size);
                Some(off)
            }
            _ => None,
        }
    }

    /// `simd_splat`: broadcast a scalar into every lane of a fresh vector slot. Integer lanes only;
    /// the string/slice SIMD search code that reaches this only uses byte vectors.
    fn emit_simd_splat(&mut self, scalar: Value, vec_ty: Type) -> Value {
        let (elem, count, es) = self.vector_info(vec_ty);
        assert!(
            !type_is_float(self.cx, elem),
            "rustc_codegen_arm64: floating-point SIMD lanes are not yet supported"
        );
        let (size, align) = self.cx.type_size_align(vec_ty);
        let off = self.alloc_slot(size, align);
        self.materialize(scalar, X10);
        let msize = mem_size(self.cx, elem);
        for i in 0..count {
            self.emit_mem_gpr(false, false, msize, X10, SP, off + i * es);
        }
        Value::Slot { off, ty: vec_ty }
    }

    /// `simd_eq`/`simd_ne`/`simd_lt`/`simd_le`/`simd_gt`/`simd_ge`: lane-wise comparison producing a
    /// mask whose lanes are all-ones where `cond` holds. Sub-word signed comparisons sign-extend the
    /// lanes first (the loads zero-extend). Integer lanes only.
    fn emit_simd_cmp(&mut self, cond: Cond, signed: bool, a: Value, b: Value, mask_ty: Type) -> Value {
        let (elem, count, es) = self.vector_info(a.ty());
        assert!(
            !type_is_float(self.cx, elem),
            "rustc_codegen_arm64: floating-point SIMD lanes are not yet supported"
        );
        let aoff = self.vector_to_slot(a);
        let boff = self.vector_to_slot(b);
        let (Some(aoff), Some(boff)) = (aoff, boff) else {
            return Value::Undef { ty: mask_ty };
        };
        let (melem, _mcount, mes) = self.vector_info(mask_ty);
        let (size, align) = self.cx.type_size_align(mask_ty);
        let roff = self.alloc_slot(size, align);
        let emsize = mem_size(self.cx, elem);
        let mmsize = mem_size(self.cx, melem);
        for i in 0..count {
            self.emit_mem_gpr(true, false, emsize, X10, SP, aoff + i * es);
            self.emit_mem_gpr(true, false, emsize, X11_HACK, SP, boff + i * es);
            if signed {
                self.sign_extend_reg(X10, es);
                self.sign_extend_reg(X11_HACK, es);
            }
            self.emit(Inst::AddSubReg {
                op: AddSub::Sub,
                size: OperandSize::S64,
                set_flags: true,
                rd: ZR,
                rn: X10,
                rm: X11_HACK,
                amount: 0,
            });
            // csetm: all-ones where `cond` holds (`csinv rd, zr, zr, !cond`).
            self.emit(Inst::CondSel {
                op: CondSel::Csinv,
                size: OperandSize::S64,
                rd: X10,
                rn: ZR,
                rm: ZR,
                cond: cond.invert(),
            });
            self.emit_mem_gpr(false, false, mmsize, X10, SP, roff + i * mes);
        }
        Value::Slot { off: roff, ty: mask_ty }
    }

    /// Sign-extend the low `es*8` bits of `reg` to 64 bits (`es` is a lane's byte width).
    fn sign_extend_reg(&mut self, reg: Gpr, es: u64) {
        let from = match es {
            1 => MemSize::B,
            2 => MemSize::H,
            4 => MemSize::W,
            _ => return,
        };
        self.emit(Inst::Sxt { from, to: OperandSize::S64, rd: reg, rn: reg });
    }

    /// `simd_and`/`simd_or`/`simd_xor`: lane-wise bitwise operation. Integer lanes only.
    fn emit_simd_binop(&mut self, op: LogicOp, a: Value, b: Value) -> Value {
        let vec_ty = a.ty();
        let (elem, count, es) = self.vector_info(vec_ty);
        let aoff = self.vector_to_slot(a);
        let boff = self.vector_to_slot(b);
        let (Some(aoff), Some(boff)) = (aoff, boff) else {
            return Value::Undef { ty: vec_ty };
        };
        let (size, align) = self.cx.type_size_align(vec_ty);
        let roff = self.alloc_slot(size, align);
        let msize = mem_size(self.cx, elem);
        for i in 0..count {
            self.emit_mem_gpr(true, false, msize, X10, SP, aoff + i * es);
            self.emit_mem_gpr(true, false, msize, X11_HACK, SP, boff + i * es);
            self.emit(Inst::Logical {
                op,
                size: OperandSize::S64,
                rd: X10,
                rn: X10,
                rm: X11_HACK,
                amount: 0,
            });
            self.emit_mem_gpr(false, false, msize, X10, SP, roff + i * es);
        }
        Value::Slot { off: roff, ty: vec_ty }
    }

    /// `simd_bitmask`: pack the most-significant bit of each lane into an integer, lane 0 in the
    /// least-significant bit (little-endian). The lanes are guaranteed to be 0 or all-ones, so the
    /// MSB test reduces to "lane is non-zero". Integer lanes only.
    fn emit_simd_bitmask(&mut self, x: Value, result_ty: Type) -> Value {
        let (elem, count, es) = self.vector_info(x.ty());
        let Some(xoff) = self.vector_to_slot(x) else {
            return Value::Undef { ty: result_ty };
        };
        let emsize = mem_size(self.cx, elem);
        self.load_imm(X10, 0, OperandSize::S64);
        for i in 0..count {
            self.emit_mem_gpr(true, false, emsize, X11_HACK, SP, xoff + i * es);
            // bit i = the lane's most-significant bit: sign-extend the lane, then test `< 0`.
            self.sign_extend_reg(X11_HACK, es);
            self.emit(Inst::AddSubImm {
                op: AddSub::Sub,
                size: OperandSize::S64,
                set_flags: true,
                rd: ZR,
                rn: X11_HACK,
                imm12: 0,
                shift12: false,
            });
            self.emit(Inst::CondSel {
                op: CondSel::Csinc,
                size: OperandSize::S64,
                rd: X11_HACK,
                rn: ZR,
                rm: ZR,
                cond: Cond::Lt.invert(),
            });
            // Accumulate `(bit << i)` into the result.
            self.emit(Inst::Logical {
                op: LogicOp::Orr,
                size: OperandSize::S64,
                rd: X10,
                rn: X10,
                rm: X11_HACK,
                amount: i as u8,
            });
        }
        self.spill(X10, result_ty)
    }

    /// `simd_reduce_all`/`simd_reduce_any`: fold every lane's truth value with AND/OR into a single
    /// boolean. Lanes are tested for non-zero (masks hold 0 or all-ones). Integer lanes only.
    fn emit_simd_reduce_bool(&mut self, x: Value, all: bool, result_ty: Type) -> Value {
        let (elem, count, es) = self.vector_info(x.ty());
        let Some(xoff) = self.vector_to_slot(x) else {
            return Value::Undef { ty: result_ty };
        };
        let emsize = mem_size(self.cx, elem);
        // `all` folds with AND from an initial `true`; `any` folds with OR from `false`.
        self.load_imm(X10, if all { 1 } else { 0 }, OperandSize::S64);
        let fold = if all { LogicOp::And } else { LogicOp::Orr };
        for i in 0..count {
            self.emit_mem_gpr(true, false, emsize, X11_HACK, SP, xoff + i * es);
            self.emit(Inst::AddSubImm {
                op: AddSub::Sub,
                size: OperandSize::S64,
                set_flags: true,
                rd: ZR,
                rn: X11_HACK,
                imm12: 0,
                shift12: false,
            });
            self.emit(Inst::CondSel {
                op: CondSel::Csinc,
                size: OperandSize::S64,
                rd: X11_HACK,
                rn: ZR,
                rm: ZR,
                cond: Cond::Eq,
            });
            self.emit(Inst::Logical {
                op: fold,
                size: OperandSize::S64,
                rd: X10,
                rn: X10,
                rm: X11_HACK,
                amount: 0,
            });
        }
        self.spill(X10, result_ty)
    }

    /// `simd_shuffle`: build a result vector by gathering lanes from the concatenation `x ++ y` at
    /// the (runtime-read) indices in `idx`. Implemented for byte lanes: the inputs are laid out
    /// contiguously in a scratch buffer and each result lane is a register-indexed byte load.
    fn emit_simd_shuffle(&mut self, x: Value, y: Value, idx: Value, result_ty: Type) -> Value {
        let (elem, n, es) = self.vector_info(x.ty());
        assert!(
            es == 1 && !type_is_float(self.cx, elem),
            "rustc_codegen_arm64: only byte-lane SIMD shuffles are supported"
        );
        let (_, out_n, _) = self.vector_info(result_ty);
        let (idx_elem, _, idx_es) = self.vector_info(idx.ty());
        let xoff = self.vector_to_slot(x);
        let yoff = self.vector_to_slot(y);
        let ioff = self.vector_to_slot(idx);
        let (Some(xoff), Some(yoff), Some(ioff)) = (xoff, yoff, ioff) else {
            return Value::Undef { ty: result_ty };
        };
        // Concatenate the inputs into a `2*n`-byte buffer so an index in `0..2*n` is a byte offset.
        let buf_off = self.alloc_slot(2 * n, 1);
        for j in 0..n {
            self.emit_mem_gpr(true, false, MemSize::B, X10, SP, xoff + j);
            self.emit_mem_gpr(false, false, MemSize::B, X10, SP, buf_off + j);
            self.emit_mem_gpr(true, false, MemSize::B, X10, SP, yoff + j);
            self.emit_mem_gpr(false, false, MemSize::B, X10, SP, buf_off + n + j);
        }
        let (out_size, out_align) = self.cx.type_size_align(result_ty);
        let res_off = self.alloc_slot(out_size, out_align);
        let idxmsize = mem_size(self.cx, idx_elem);
        for i in 0..out_n {
            self.emit_mem_gpr(true, false, idxmsize, X12, SP, ioff + i * idx_es);
            self.emit_frame_addr(X13, buf_off);
            self.emit(Inst::AddSubReg {
                op: AddSub::Add,
                size: OperandSize::S64,
                set_flags: false,
                rd: X13,
                rn: X13,
                rm: X12,
                amount: 0,
            });
            self.emit(Inst::LoadStoreUImm {
                load: true,
                signed: false,
                size: MemSize::B,
                rt: X10,
                rn: X13,
                offset: 0,
            });
            self.emit_mem_gpr(false, false, MemSize::B, X10, SP, res_off + i);
        }
        Value::Slot { off: res_off, ty: result_ty }
    }

    /// `simd_extract`: read a single lane (at the constant index `idx`) out of a vector.
    fn emit_simd_extract(&mut self, x: Value, idx: u64, result_ty: Type) -> Value {
        let (elem, _count, es) = self.vector_info(x.ty());
        let Some(xoff) = self.vector_to_slot(x) else {
            return Value::Undef { ty: result_ty };
        };
        let lane_off = xoff + idx * es;
        if type_is_float(self.cx, elem) {
            self.emit_mem_fp(true, fp_size(self.cx, elem), V16, SP, lane_off);
            self.spill_fp(V16, result_ty)
        } else {
            self.emit_mem_gpr(true, false, mem_size(self.cx, elem), X10, SP, lane_off);
            self.spill(X10, result_ty)
        }
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

/// The incoming physical location of a parameter in the baseline AAPCS64 ABI.
enum ParamLoc {
    /// Integer/pointer argument in `x0..x7` (or the indirect-return pointer in `x8`).
    Gpr(Gpr),
    /// A 128-bit integer argument occupying two consecutive integer registers (low, high).
    GprPair(Gpr, Gpr),
    /// Floating-point argument in `v0..v7`.
    Fp(Vreg),
    /// Argument passed on the stack at the given byte offset within the caller's outgoing area.
    /// The callee reads it at `fp + 16 + offset`; the caller writes it at `sp + offset`.
    Stack(u32),
}

/// AAPCS64 argument-assignment cursor: the next free integer register (`x{ngrn}`), the next free
/// floating-point register (`v{nsrn}`), and the next free byte offset on the stack (`nsaa`).
#[derive(Default)]
struct ArgAssign {
    ngrn: u8,
    nsrn: u8,
    nsaa: u32,
}

/// The classified parameter locations of a function signature, plus the total bytes of stack
/// arguments — which, for a *call* to such a function, is exactly its outgoing-argument-area size.
struct ParamList {
    params: Vec<(ParamLoc, Type)>,
    stack_size: u32,
}

/// Assign one scalar parameter to its next location: a register from the appropriate bank, or a
/// stack slot once that bank is exhausted.
fn push_scalar_param(cx: &CodegenCx<'_>, params: &mut Vec<(ParamLoc, Type)>, a: &mut ArgAssign, ty: Type) {
    // A 128-bit integer takes two *consecutive* integer registers (the Apple AArch64 ABI does not
    // even-align them); if fewer than two remain it is passed on the stack, 16-byte aligned, and the
    // remaining single register is left unused.
    if matches!(cx.type_data(ty), TypeData::Int(b) if b > 64) {
        if a.ngrn <= 6 {
            let lo = Gpr::from_encoding(a.ngrn);
            let hi = Gpr::from_encoding(a.ngrn + 1);
            a.ngrn += 2;
            params.push((ParamLoc::GprPair(lo, hi), ty));
        } else {
            a.ngrn = 8;
            a.nsaa = (a.nsaa + 15) & !15;
            let off = a.nsaa;
            a.nsaa += 16;
            params.push((ParamLoc::Stack(off), ty));
        }
        return;
    }
    let loc = if type_is_float(cx, ty) {
        if a.nsrn < 8 {
            let loc = ParamLoc::Fp(Vreg::from_encoding(a.nsrn));
            a.nsrn += 1;
            loc
        } else {
            let loc = ParamLoc::Stack(a.nsaa);
            a.nsaa += 8;
            loc
        }
    } else if a.ngrn < 8 {
        let loc = ParamLoc::Gpr(Gpr::from_encoding(a.ngrn));
        a.ngrn += 1;
        loc
    } else {
        let loc = ParamLoc::Stack(a.nsaa);
        a.nsaa += 8;
        loc
    };
    params.push((loc, ty));
}

/// Compute the location and backend type of each physical parameter, following the baseline
/// AAPCS64 split: integer/pointer arguments fill `x0..x7`, floating-point arguments fill `v0..v7`,
/// an indirect-return (sret) pointer arrives in `x8`, and anything left once a register bank is
/// exhausted is passed on the stack (each scalar occupying an 8-byte slot).
fn build_param_list<'tcx>(cx: &CodegenCx<'tcx>, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> ParamList {
    let mut params = Vec::new();
    let ptr = cx.intern_type(TypeData::Ptr);
    if fn_abi.ret.is_indirect() {
        // The indirect-return (sret) pointer arrives in x8 (never on the stack).
        params.push((ParamLoc::Gpr(Gpr::from_encoding(8)), ptr));
    }
    let mut a = ArgAssign::default();
    for arg in fn_abi.args.iter() {
        match arg.mode {
            PassMode::Ignore => {}
            PassMode::Direct(_) => {
                let ty = cx.immediate_backend_type(arg.layout);
                push_scalar_param(cx, &mut params, &mut a, ty);
            }
            PassMode::Pair(..) => {
                let x = cx.scalar_pair_element_backend_type(arg.layout, 0, true);
                let y = cx.scalar_pair_element_backend_type(arg.layout, 1, true);
                push_scalar_param(cx, &mut params, &mut a, x);
                push_scalar_param(cx, &mut params, &mut a, y);
            }
            PassMode::Indirect { .. } => push_scalar_param(cx, &mut params, &mut a, ptr),
            PassMode::Cast { .. } => {
                push_scalar_param(cx, &mut params, &mut a, cx.intern_type(TypeData::Int(64)))
            }
        }
    }
    ParamList { params, stack_size: a.nsaa }
}

/// Compute the size of the outgoing-argument area for `instance`: the maximum, over every call in
/// its MIR body, of the bytes needed for arguments that do not fit in registers. This is done once
/// up front (before instruction selection) so that local slots can be laid out above the area and
/// every stack reference emitted with its final offset, without a second pass over the instructions.
fn outgoing_arg_bytes<'tcx>(cx: &CodegenCx<'tcx>, instance: Instance<'tcx>) -> u32 {
    let tcx = cx.tcx;
    let typing_env = cx.typing_env();
    let mir = tcx.instance_mir(instance.def);
    let mut max = 0u32;
    for bb in mir.basic_blocks.iter() {
        let Some(terminator) = bb.terminator.as_ref() else { continue };
        let rustc_middle::mir::TerminatorKind::Call { func, args, .. } = &terminator.kind else {
            continue;
        };
        // Monomorphize the callee type (and any C-variadic extra arguments) in this instance.
        let callee_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            typing_env,
            ty::EarlyBinder::bind(tcx, func.ty(mir, tcx)),
        );
        let sig = callee_ty.fn_sig(tcx);
        let n_fixed = sig.inputs().skip_binder().len().min(args.len());
        let extra_args = tcx.mk_type_list_from_iter(args[n_fixed..].iter().map(|arg| {
            instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                typing_env,
                ty::EarlyBinder::bind(tcx, arg.node.ty(mir, tcx)),
            )
        }));
        let fn_abi = match *callee_ty.kind() {
            // A direct call to a resolvable free/associated function: use the instance ABI, which
            // accounts for `#[track_caller]`'s implicit caller-location argument.
            ty::FnDef(def_id, generic_args)
                if matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn) =>
            {
                match ty::Instance::try_resolve(tcx, typing_env, def_id, generic_args) {
                    Ok(Some(callee)) => cx.fn_abi_of_instance(callee, extra_args),
                    _ => cx.fn_abi_of_fn_ptr(sig, extra_args),
                }
            }
            // Tuple-struct/enum constructors and indirect calls: the signature determines the ABI.
            ty::FnDef(..) | ty::FnPtr(..) => cx.fn_abi_of_fn_ptr(sig, extra_args),
            _ => continue,
        };
        max = max.max(build_param_list(cx, fn_abi).stack_size);
    }
    max
}

/// Spill each incoming parameter register into a frame slot at function entry and record the slots.
fn setup_params(cx: &CodegenCx<'_>, fb: &mut FunctionBuild, block: BasicBlock) {
    let params = match cx.cur_instance.get() {
        Some(instance) => {
            let fn_abi = cx.fn_abi_of_instance(instance, ty::List::empty());
            build_param_list(cx, fn_abi).params
        }
        // The synthesized C `main` entry wrapper has no MIR instance. Where `main` is
        // `int main(int argc, char** argv)`, spill those two incoming registers so the generic
        // entry-wrapper builder can read them back via `get_param`.
        None if cx.tcx.sess.target.main_needs_argc_argv => vec![
            (ParamLoc::Gpr(Gpr::from_encoding(0)), cx.intern_type(TypeData::Int(32))),
            (ParamLoc::Gpr(Gpr::from_encoding(1)), cx.intern_type(TypeData::Ptr)),
        ],
        None => return,
    };
    for (loc, ty) in params {
        let (size, align) = cx.type_size_align(ty);
        let off = fb.frame.alloc_local(size, align);
        // Move the incoming parameter into its frame slot. Register parameters are stored directly;
        // a stack parameter is first loaded from the caller-provided incoming area (`fp + 16 + k`)
        // through a scratch register, then stored into its slot.
        match loc {
            ParamLoc::Gpr(reg) => {
                let out = &mut fb.blocks[block.0 as usize];
                push_mem_gpr(out, false, false, mem_size(cx, ty), reg, SP, off);
            }
            ParamLoc::GprPair(lo, hi) => {
                // A 128-bit integer arrives in two consecutive registers; store both words.
                let out = &mut fb.blocks[block.0 as usize];
                push_mem_gpr(out, false, false, MemSize::X, lo, SP, off);
                push_mem_gpr(out, false, false, MemSize::X, hi, SP, off + 8);
            }
            ParamLoc::Fp(reg) => {
                let out = &mut fb.blocks[block.0 as usize];
                push_mem_fp(out, false, fp_size(cx, ty), reg, SP, off);
            }
            ParamLoc::Stack(arg_off) => {
                let incoming = fb.frame.incoming_arg(arg_off as u64);
                let out = &mut fb.blocks[block.0 as usize];
                if matches!(cx.type_data(ty), TypeData::Int(b) if b > 64) {
                    // A 128-bit integer on the stack: copy both incoming words into the slot.
                    push_mem_gpr(out, true, false, MemSize::X, X9, FP, incoming);
                    push_mem_gpr(out, false, false, MemSize::X, X9, SP, off);
                    push_mem_gpr(out, true, false, MemSize::X, X9, FP, incoming + 8);
                    push_mem_gpr(out, false, false, MemSize::X, X9, SP, off + 8);
                } else if type_is_float(cx, ty) {
                    let size = fp_size(cx, ty);
                    push_mem_fp(out, true, size, V16, FP, incoming);
                    push_mem_fp(out, false, size, V16, SP, off);
                } else {
                    let size = mem_size(cx, ty);
                    push_mem_gpr(out, true, false, size, X9, FP, incoming);
                    push_mem_gpr(out, false, false, size, X9, SP, off);
                }
            }
        }
        fb.param_slots.push(Value::Slot { off, ty });
    }
}

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Load a value into `reg`, emitting whatever is needed (immediate move, frame load, or address
    /// formation).
    fn materialize(&mut self, val: Value, reg: Gpr) {
        debug_assert!(
            !matches!(self.cx.type_data(val.ty()), TypeData::Vector(..)),
            "a vector value cannot be materialized into a scalar register; it must be processed \
             lane by lane from its frame slot"
        );
        // A 128-bit integer does not fit in one register and must be handled as a low/high word
        // pair (see `materialize128`). Reaching here with one means an i128 operation was not routed
        // to its dedicated path; fail loudly rather than silently load only the low 64 bits.
        assert!(
            !matches!(self.cx.type_data(val.ty()), TypeData::Int(b) if b > 64),
            "rustc_codegen_arm64: a 128-bit value cannot be materialized into a single register"
        );
        match val {
            Value::Const { bits, ty } => self.load_imm(reg, bits, op_size(self.cx, ty)),
            Value::Undef { ty } => self.load_imm(reg, 0, op_size(self.cx, ty)),
            Value::Slot { off, ty } => {
                self.emit_mem_gpr(true, false, mem_size(self.cx, ty), reg, SP, off);
            }
            Value::Sym { sym, offset, ty: _ } => {
                let symref = SymRef { name: self.cx.sym_name(sym), addend: offset };
                self.emit(Inst::Adrp { rd: reg, sym: symref.clone() });
                self.emit(Inst::AddLo { rd: reg, rn: reg, sym: symref });
            }
        }
    }

    /// Load a value into `reg`, sign-extending sub-word integers (`i8`/`i16`) up to their operation
    /// width. The default [`materialize`](Self::materialize) zero-extends (e.g. `ldrb`), which is
    /// correct for width-insensitive ops but wrong for signed compares, divides, and arithmetic
    /// shifts: there the high bits of the register must carry the sign. Wider integers already span
    /// their operation width, so they are loaded unchanged.
    fn materialize_signed(&mut self, val: Value, reg: Gpr) {
        self.materialize(val, reg);
        let from = match self.cx.type_data(val.ty()) {
            TypeData::Int(8) => MemSize::B,
            TypeData::Int(16) => MemSize::H,
            _ => return,
        };
        let to = op_size(self.cx, val.ty());
        self.emit(Inst::Sxt { from, to, rd: reg, rn: reg });
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
        let off = self.alloc_slot(size, align);
        self.emit_mem_gpr(false, false, mem_size(self.cx, ty), reg, SP, off);
        Value::Slot { off, ty }
    }

    /// Whether `ty` is a 128-bit integer (handled as a low/high 64-bit word pair).
    fn is_int128(&self, ty: Type) -> bool {
        matches!(self.cx.type_data(ty), TypeData::Int(b) if b > 64)
    }

    /// Load a 128-bit value into the register pair `(lo, hi)` (low word in `lo`, high in `hi`).
    fn materialize128(&mut self, val: Value, lo: Gpr, hi: Gpr) {
        match val {
            Value::Const { bits, .. } => {
                self.load_imm(lo, bits, OperandSize::S64);
                self.load_imm(hi, bits >> 64, OperandSize::S64);
            }
            Value::Undef { .. } => {
                self.load_imm(lo, 0, OperandSize::S64);
                self.load_imm(hi, 0, OperandSize::S64);
            }
            Value::Slot { off, .. } => {
                self.emit_mem_gpr(true, false, MemSize::X, lo, SP, off);
                self.emit_mem_gpr(true, false, MemSize::X, hi, SP, off + 8);
            }
            Value::Sym { .. } => {
                panic!("rustc_codegen_arm64: a symbol address cannot be a 128-bit integer value")
            }
        }
    }

    /// Store the register pair `(lo, hi)` into a fresh 16-byte frame slot and return it.
    fn spill128(&mut self, lo: Gpr, hi: Gpr, ty: Type) -> Value {
        let (size, align) = self.cx.type_size_align(ty);
        let off = self.alloc_slot(size, align);
        self.emit_mem_gpr(false, false, MemSize::X, lo, SP, off);
        self.emit_mem_gpr(false, false, MemSize::X, hi, SP, off + 8);
        Value::Slot { off, ty }
    }

    /// Emit a carry-chained 128-bit add or subtract: `(lo, hi) op= (b_lo, b_hi)`. With `set_flags`,
    /// the final `NZCV` reflects the full 128-bit result (used for overflow detection).
    fn int128_addsub(&mut self, op: AddSub, lhs: Value, rhs: Value, set_flags: bool) -> Value {
        let ty = lhs.ty();
        self.materialize128(lhs, X9, X10);
        self.materialize128(rhs, X11, X12);
        // Low word sets the carry; high word consumes it.
        self.emit(Inst::AddSubReg { op, size: OperandSize::S64, set_flags: true, rd: X9, rn: X9, rm: X11, amount: 0 });
        self.emit(Inst::AddSubCarry { op, size: OperandSize::S64, set_flags, rd: X10, rn: X10, rm: X12 });
        self.spill128(X9, X10, ty)
    }

    /// Emit a per-word 128-bit bitwise logical op (`and`/`orr`/`eor`).
    fn int128_logical(&mut self, op: LogicOp, lhs: Value, rhs: Value) -> Value {
        let ty = lhs.ty();
        self.materialize128(lhs, X9, X10);
        self.materialize128(rhs, X11, X12);
        self.emit(Inst::Logical { op, size: OperandSize::S64, rd: X9, rn: X9, rm: X11, amount: 0 });
        self.emit(Inst::Logical { op, size: OperandSize::S64, rd: X10, rn: X10, rm: X12, amount: 0 });
        self.spill128(X9, X10, ty)
    }

    /// 128 x 128 -> low 128-bit multiply, inline via 64-bit partial products (no libcall):
    /// `result_lo = a_lo*b_lo`, `result_hi = hi64(a_lo*b_lo) + a_hi*b_lo + a_lo*b_hi`.
    fn int128_mul(&mut self, lhs: Value, rhs: Value) -> Value {
        let ty = lhs.ty();
        self.materialize128(lhs, X9, X10); // a = (a_lo=X9, a_hi=X10)
        self.materialize128(rhs, X11, X12); // b = (b_lo=X11, b_hi=X12)
        // X13 = hi64(a_lo * b_lo)
        self.emit(Inst::MulHigh { signed: false, rd: X13, rn: X9, rm: X11 });
        // X13 += a_hi * b_lo
        self.emit(Inst::Madd { size: OperandSize::S64, rd: X13, rn: X10, rm: X11, ra: X13 });
        // result_hi (X13) = a_lo * b_hi + X13
        self.emit(Inst::Madd { size: OperandSize::S64, rd: X13, rn: X9, rm: X12, ra: X13 });
        // result_lo (X9) = a_lo * b_lo
        self.emit(Inst::Madd { size: OperandSize::S64, rd: X9, rn: X9, rm: X11, ra: ZR });
        self.spill128(X9, X13, ty)
    }

    /// A 128-bit comparison returning an `i1`. Equality compares both words and folds with `orr`;
    /// ordered comparisons do a full 128-bit subtract (`subs`/`sbcs`) and read the flags, swapping
    /// operands for the `>`/`<=` forms (see the per-case table in `icmp`).
    fn int128_icmp(&mut self, op: IntPredicate, lhs: Value, rhs: Value) -> Value {
        let bool_ty = self.cx.intern_type(TypeData::Int(1));
        if matches!(op, IntPredicate::IntEQ | IntPredicate::IntNE) {
            self.materialize128(lhs, X9, X10);
            self.materialize128(rhs, X11, X12);
            self.emit(Inst::Logical { op: LogicOp::Eor, size: OperandSize::S64, rd: X9, rn: X9, rm: X11, amount: 0 });
            self.emit(Inst::Logical { op: LogicOp::Eor, size: OperandSize::S64, rd: X10, rn: X10, rm: X12, amount: 0 });
            self.emit(Inst::Logical { op: LogicOp::Orr, size: OperandSize::S64, rd: X9, rn: X9, rm: X10, amount: 0 });
            self.emit(Inst::AddSubImm { op: AddSub::Sub, size: OperandSize::S64, set_flags: true, rd: ZR, rn: X9, imm12: 0, shift12: false });
            let cond = if matches!(op, IntPredicate::IntEQ) { Cond::Eq } else { Cond::Ne };
            self.emit(Inst::CondSel { op: CondSel::Csinc, size: OperandSize::S32, rd: X9, rn: ZR, rm: ZR, cond: cond.invert() });
            return self.spill(X9, bool_ty);
        }
        // Ordered: (first, second, cond). `>`/`<=` swap operands so all four reduce to lt/ge tests.
        let (swap, cond) = match op {
            IntPredicate::IntULT => (false, Cond::Lo),
            IntPredicate::IntSLT => (false, Cond::Lt),
            IntPredicate::IntUGE => (false, Cond::Hs),
            IntPredicate::IntSGE => (false, Cond::Ge),
            IntPredicate::IntUGT => (true, Cond::Lo),
            IntPredicate::IntSGT => (true, Cond::Lt),
            IntPredicate::IntULE => (true, Cond::Hs),
            IntPredicate::IntSLE => (true, Cond::Ge),
            IntPredicate::IntEQ | IntPredicate::IntNE => unreachable!(),
        };
        let (first, second) = if swap { (rhs, lhs) } else { (lhs, rhs) };
        self.materialize128(first, X9, X10);
        self.materialize128(second, X11, X12);
        self.emit(Inst::AddSubReg { op: AddSub::Sub, size: OperandSize::S64, set_flags: true, rd: ZR, rn: X9, rm: X11, amount: 0 });
        self.emit(Inst::AddSubCarry { op: AddSub::Sub, size: OperandSize::S64, set_flags: true, rd: ZR, rn: X10, rm: X12 });
        self.emit(Inst::CondSel { op: CondSel::Csinc, size: OperandSize::S32, rd: X9, rn: ZR, rm: ZR, cond: cond.invert() });
        self.spill(X9, bool_ty)
    }

    /// A 128-bit binary op implemented by a compiler-builtins libcall taking two `i128` arguments in
    /// the `x0:x1` and `x2:x3` register pairs and returning an `i128` in `x0:x1` (division and
    /// remainder: `__udivti3`/`__divti3`/`__umodti3`/`__modti3`).
    fn int128_bin_libcall(&mut self, sym: &str, lhs: Value, rhs: Value) -> Value {
        let ty = lhs.ty();
        self.materialize128(lhs, X0, X1);
        self.materialize128(rhs, X2, X3);
        self.emit(Inst::Bl { sym: SymRef::new(sym) });
        self.spill128(X0, X1, ty)
    }

    /// A 128-bit shift implemented by a compiler-builtins libcall (`__ashlti3`/`__lshrti3`/
    /// `__ashrti3`): the value is in `x0:x1`, the count (already masked to `< 128` by the generic
    /// codegen) in `w2`, and the result returns in `x0:x1`.
    fn int128_shift(&mut self, sym: &str, lhs: Value, rhs: Value) -> Value {
        let ty = lhs.ty();
        self.materialize128(lhs, X0, X1);
        if self.is_int128(rhs.ty()) {
            // The shift amount arrives as an i128 (extended to the value's type); its low word is
            // the count.
            self.materialize128(rhs, X2, X9);
        } else {
            self.materialize(rhs, X2);
        }
        self.emit(Inst::Bl { sym: SymRef::new(sym) });
        self.spill128(X0, X1, ty)
    }

    /// Convert a 128-bit integer to a float via a compiler-builtins libcall (`__float[un]tidf` /
    /// `__float[un]tisf`): the integer is in `x0:x1`, the result returns in `d0`/`s0`.
    fn int128_to_fp(&mut self, signed: bool, val: Value, dest_ty: Type) -> Value {
        let sym = match (fp_size(self.cx, dest_ty), signed) {
            (FpSize::S64, true) => "___floattidf",
            (FpSize::S64, false) => "___floatuntidf",
            (FpSize::S32, true) => "___floattisf",
            (FpSize::S32, false) => "___floatuntisf",
        };
        self.materialize128(val, X0, X1);
        self.emit(Inst::Bl { sym: SymRef::new(sym) });
        self.spill_fp(V0, dest_ty)
    }

    /// Convert a float to a 128-bit integer via a compiler-builtins libcall (`__fix[uns]dfti` /
    /// `__fix[uns]sfti`): the float is in `d0`/`s0`, the result returns in `x0:x1`. These saturate
    /// out-of-range inputs and map NaN to zero, matching Rust's `as` semantics.
    fn fp_to_int128(&mut self, signed: bool, val: Value, dest_ty: Type) -> Value {
        let sym = match (fp_size(self.cx, val.ty()), signed) {
            (FpSize::S64, true) => "___fixdfti",
            (FpSize::S64, false) => "___fixunsdfti",
            (FpSize::S32, true) => "___fixsfti",
            (FpSize::S32, false) => "___fixunssfti",
        };
        self.materialize_fp(val, V0);
        self.emit(Inst::Bl { sym: SymRef::new(sym) });
        self.spill128(X0, X1, dest_ty)
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

    /// Reverse bits (`rbit`) or bytes (`rev`) of a value within its own width. The hardware op
    /// reverses the whole 32/64-bit register, so for a sub-word type the reversed low-`N` bits land
    /// in the register's top end and are shifted back down by `op_bits - N`.
    fn emit_reverse(&mut self, v: Value, op: DataProc1) -> Value {
        let (n, op_bits) = match self.cx.type_data(v.ty()) {
            TypeData::Int(b) if b <= 32 => (b, 32u32),
            TypeData::Int(b) => (b, 64),
            _ => (64, 64),
        };
        let size = op_size(self.cx, v.ty());
        self.materialize(v, X9);
        self.emit(Inst::DataProc1 { op, size, rd: X9, rn: X9 });
        if n < op_bits {
            self.load_imm(X10, (op_bits - n) as u128, size);
            self.emit(Inst::DataProc2 { op: DataProc2::Lsrv, size, rd: X9, rn: X9, rm: X10 });
        }
        self.spill(X9, v.ty())
    }

    /// Clamp the result of a float-to-int conversion (in `raw_reg`, saturated by the hardware only
    /// to the 32/64-bit operation width) into the destination type's own range, then narrow to it.
    /// AArch64's `fcvtzs`/`fcvtzu` saturate to the 32- or 64-bit register, so for an `i8`/`i16`/etc.
    /// destination the out-of-range value must be additionally clamped (e.g. `300.0 as u8 == 255`,
    /// not `44`). Returns `raw_reg` spilled directly when the destination already spans the op width.
    fn clamp_fp_to_int(&mut self, raw_reg: Gpr, dest_ty: Type, signed: bool) -> Value {
        let n = match self.cx.type_data(dest_ty) {
            TypeData::Int(b) => b,
            _ => return self.spill(raw_reg, dest_ty),
        };
        let op_bits = match op_size(self.cx, dest_ty) {
            OperandSize::S32 => 32,
            OperandSize::S64 => 64,
        };
        if n >= op_bits {
            return self.spill(raw_reg, dest_ty);
        }
        let op_ty = self.cx.intern_type(TypeData::Int(op_bits));
        let raw = self.spill(raw_reg, op_ty);
        let clamped = if signed {
            let max = self.cx.const_uint(op_ty, ((1i128 << (n - 1)) - 1) as u64);
            let min = self.cx.const_uint(op_ty, (-(1i128 << (n - 1))) as i64 as u64);
            let too_high = self.icmp(IntPredicate::IntSGT, raw, max);
            let raw = self.select(too_high, max, raw);
            let too_low = self.icmp(IntPredicate::IntSLT, raw, min);
            self.select(too_low, min, raw)
        } else {
            let max = self.cx.const_uint(op_ty, ((1u128 << n) - 1) as u64);
            let too_high = self.icmp(IntPredicate::IntUGT, raw, max);
            self.select(too_high, max, raw)
        };
        self.trunc(clamped, dest_ty)
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
                self.emit_mem_fp(true, size, vreg, SP, off);
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
        let off = self.alloc_slot(size, align);
        self.emit_mem_fp(false, fp_size(self.cx, ty), vreg, SP, off);
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

    /// Emit a one-operand FP instruction (`fabs`/`fsqrt`/`frint*`): materialize the operand into
    /// `v16`, compute in place, and spill.
    fn fp_unary(&mut self, op: FpOp1, arg: Value) -> Value {
        let ty = arg.ty();
        let size = fp_size(self.cx, ty);
        self.materialize_fp(arg, V16);
        self.emit(Inst::FpDataProc1 { op, size, rd: V16, rn: V16 });
        self.spill_fp(V16, ty)
    }

    /// Emit a fused multiply-add (`a * b + c`, single rounding) via `fmadd`.
    fn fp_fma(&mut self, a: Value, b: Value, c: Value) -> Value {
        let ty = a.ty();
        let size = fp_size(self.cx, ty);
        self.materialize_fp(a, V16);
        self.materialize_fp(b, V17);
        self.materialize_fp(c, V0);
        self.emit(Inst::FpFma { size, rd: V16, rn: V16, rm: V17, ra: V0 });
        self.spill_fp(V16, ty)
    }

    /// The C library symbol for a libm routine: the `f64` form is `base` and the `f32` form
    /// appends `f` (e.g. `sin`/`sinf`), with the Mach-O `_` prefix.
    fn libm_symbol(&self, base: &str, ty: Type) -> String {
        match fp_size(self.cx, ty) {
            FpSize::S64 => format!("_{base}"),
            FpSize::S32 => format!("_{base}f"),
        }
    }

    /// Call a unary libm routine, passing the argument in `d0`/`s0` and collecting the result.
    fn fp_libm_unary(&mut self, base: &str, arg: Value) -> Value {
        let ty = arg.ty();
        let sym = self.libm_symbol(base, ty);
        self.materialize_fp(arg, V0);
        self.emit(Inst::Bl { sym: SymRef::new(sym) });
        self.spill_fp(V0, ty)
    }

    /// Call a binary libm routine (`pow`, `copysign`), passing args in `d0`/`d1` (or `s0`/`s1`).
    fn fp_libm_binary(&mut self, base: &str, a: Value, b: Value) -> Value {
        let ty = a.ty();
        let sym = self.libm_symbol(base, ty);
        self.materialize_fp(a, V0);
        self.materialize_fp(b, V1);
        self.emit(Inst::Bl { sym: SymRef::new(sym) });
        self.spill_fp(V0, ty)
    }

    /// Integer power (`powi`): the base is passed in `d0`/`s0` and the `i32` exponent in `w0`,
    /// calling the compiler-builtins routine `__powidf2`/`__powisf2`.
    fn fp_powi(&mut self, base_val: Value, exp: Value) -> Value {
        let ty = base_val.ty();
        let sym = match fp_size(self.cx, ty) {
            FpSize::S64 => "___powidf2",
            FpSize::S32 => "___powisf2",
        };
        self.materialize_fp(base_val, V0);
        self.materialize(exp, X0);
        self.emit(Inst::Bl { sym: SymRef::new(sym) });
        self.spill_fp(V0, ty)
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

    /// Like [`alu_rrr`](Self::alu_rrr) but sign-extends sub-word operands first, for operations
    /// whose result depends on the operands' sign (e.g. signed division).
    fn alu_rrr_signed(
        &mut self,
        mk: impl Fn(OperandSize, Gpr, Gpr, Gpr) -> Inst,
        lhs: Value,
        rhs: Value,
    ) -> Value {
        let ty = lhs.ty();
        let size = op_size(self.cx, ty);
        self.materialize_signed(lhs, X9);
        self.materialize_signed(rhs, X10);
        self.emit(mk(size, X9, X9, X10));
        self.spill(X9, ty)
    }

    /// Emit `cmp`/`cset` for an integer comparison, returning an `i1` value.
    fn emit_icmp(&mut self, cond: Cond, lhs: Value, rhs: Value) -> Value {
        let size = op_size(self.cx, lhs.ty());
        // Signed comparisons require the operands sign-extended to the compare width; unsigned and
        // equality comparisons are already correct with the default zero-extending load.
        if matches!(cond, Cond::Lt | Cond::Le | Cond::Gt | Cond::Ge) {
            self.materialize_signed(lhs, X9);
            self.materialize_signed(rhs, X10);
        } else {
            self.materialize(lhs, X9);
            self.materialize(rhs, X10);
        }
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
            // Byte reverse (`swap_bytes`) and bit reverse (`reverse_bits`).
            sym::bswap => {
                IntrinsicResult::Operand(OperandValue::Immediate(
                    self.emit_reverse(args[0].immediate(), DataProc1::Rev),
                ))
            }
            sym::bitreverse => {
                IntrinsicResult::Operand(OperandValue::Immediate(
                    self.emit_reverse(args[0].immediate(), DataProc1::Rbit),
                ))
            }
            // `compare_bytes` has `memcmp` semantics: the sign of the first differing byte, as i32.
            sym::compare_bytes => {
                self.materialize(args[0].immediate(), X0);
                self.materialize(args[1].immediate(), X1);
                self.materialize(args[2].immediate(), X2);
                self.emit(Inst::Bl { sym: SymRef::new("_memcmp") });
                let result_ty = self.cx.immediate_backend_type(result_layout);
                IntrinsicResult::Operand(OperandValue::Immediate(self.spill(X0, result_ty)))
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
            // Floating-point math intrinsics that map to a single ARM64 instruction with identical
            // IEEE-754 semantics: square root, and the directed/round-to-nearest rounding modes.
            sym::sqrtf32 | sym::sqrtf64 => {
                let r = self.fp_unary(FpOp1::Fsqrt, args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::floorf32 | sym::floorf64 => {
                let r = self.fp_unary(FpOp1::Frintm, args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::ceilf32 | sym::ceilf64 => {
                let r = self.fp_unary(FpOp1::Frintp, args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::truncf32 | sym::truncf64 => {
                let r = self.fp_unary(FpOp1::Frintz, args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            // `round` is ties-away-from-zero (`frinta`); `round_ties_even` is ties-to-even (`frintn`).
            sym::roundf32 | sym::roundf64 => {
                let r = self.fp_unary(FpOp1::Frinta, args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::round_ties_even_f32 | sym::round_ties_even_f64 => {
                let r = self.fp_unary(FpOp1::Frintn, args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            // `fabs` is generic over the float type; f32/f64 map to `fabs`, wider/narrower floats
            // are unsupported and fall through to the loud "must be overridden" error.
            sym::fabs => {
                let arg = args[0].immediate();
                if matches!(self.cx.type_data(arg.ty()), TypeData::Float(32) | TypeData::Float(64)) {
                    let r = self.fp_unary(FpOp1::Fabs, arg);
                    IntrinsicResult::Operand(OperandValue::Immediate(r))
                } else {
                    IntrinsicResult::Fallback(instance)
                }
            }
            // Fused multiply-add (`fma`, single rounding) and `fmuladd` (fusing permitted) -> `fmadd`.
            sym::fmaf32 | sym::fmaf64 | sym::fmuladdf32 | sym::fmuladdf64 => {
                let r =
                    self.fp_fma(args[0].immediate(), args[1].immediate(), args[2].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            // The transcendental routines and `copysign`/`pow` have no exact single-instruction
            // form, so they call the corresponding libm routine (always linked in `libSystem`).
            sym::sinf32 | sym::sinf64 => {
                let r = self.fp_libm_unary("sin", args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::cosf32 | sym::cosf64 => {
                let r = self.fp_libm_unary("cos", args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::expf32 | sym::expf64 => {
                let r = self.fp_libm_unary("exp", args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::exp2f32 | sym::exp2f64 => {
                let r = self.fp_libm_unary("exp2", args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::logf32 | sym::logf64 => {
                let r = self.fp_libm_unary("log", args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::log2f32 | sym::log2f64 => {
                let r = self.fp_libm_unary("log2", args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::log10f32 | sym::log10f64 => {
                let r = self.fp_libm_unary("log10", args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::powf32 | sym::powf64 => {
                let r = self.fp_libm_binary("pow", args[0].immediate(), args[1].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::copysignf32 | sym::copysignf64 => {
                let r = self.fp_libm_binary("copysign", args[0].immediate(), args[1].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            // Integer power -> the compiler-builtins routine `__powidf2`/`__powisf2`.
            sym::powif32 | sym::powif64 => {
                let r = self.fp_powi(args[0].immediate(), args[1].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            // Integer-lane SIMD intrinsics, lowered scalarly (lane by lane in memory). These are
            // reached by the portable-SIMD substring/slice search in `core` (`u8x16`/`mask8x16`).
            sym::simd_splat => {
                let vec_ty = self.cx.immediate_backend_type(result_layout);
                let r = self.emit_simd_splat(args[0].immediate(), vec_ty);
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::simd_eq
            | sym::simd_ne
            | sym::simd_lt
            | sym::simd_le
            | sym::simd_gt
            | sym::simd_ge => {
                // The comparison's signedness comes from the lane element type (the backend `Int`
                // type does not record it).
                let signed = args[0]
                    .layout
                    .ty
                    .simd_size_and_type(self.cx.tcx)
                    .1
                    .is_signed();
                let pred = match name {
                    sym::simd_eq => IntPredicate::IntEQ,
                    sym::simd_ne => IntPredicate::IntNE,
                    sym::simd_lt if signed => IntPredicate::IntSLT,
                    sym::simd_lt => IntPredicate::IntULT,
                    sym::simd_le if signed => IntPredicate::IntSLE,
                    sym::simd_le => IntPredicate::IntULE,
                    sym::simd_gt if signed => IntPredicate::IntSGT,
                    sym::simd_gt => IntPredicate::IntUGT,
                    sym::simd_ge if signed => IntPredicate::IntSGE,
                    _ => IntPredicate::IntUGE,
                };
                let cond = int_pred_to_cond(pred);
                let mask_ty = self.cx.immediate_backend_type(result_layout);
                let r = self.emit_simd_cmp(
                    cond,
                    signed,
                    args[0].immediate(),
                    args[1].immediate(),
                    mask_ty,
                );
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::simd_and => {
                let r = self.emit_simd_binop(LogicOp::And, args[0].immediate(), args[1].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::simd_or => {
                let r = self.emit_simd_binop(LogicOp::Orr, args[0].immediate(), args[1].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::simd_xor => {
                let r = self.emit_simd_binop(LogicOp::Eor, args[0].immediate(), args[1].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::simd_bitmask => {
                let result_ty = self.cx.immediate_backend_type(result_layout);
                let r = self.emit_simd_bitmask(args[0].immediate(), result_ty);
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::simd_reduce_all | sym::simd_reduce_any => {
                let result_ty = self.cx.immediate_backend_type(result_layout);
                let all = name == sym::simd_reduce_all;
                let r = self.emit_simd_reduce_bool(args[0].immediate(), all, result_ty);
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::simd_shuffle => {
                let result_ty = self.cx.immediate_backend_type(result_layout);
                let r = self.emit_simd_shuffle(
                    args[0].immediate(),
                    args[1].immediate(),
                    args[2].immediate(),
                    result_ty,
                );
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::simd_extract => {
                let idx = match args[1].immediate() {
                    Value::Const { bits, .. } => bits as u64,
                    _ => 0,
                };
                let result_ty = self.cx.immediate_backend_type(result_layout);
                let r = self.emit_simd_extract(args[0].immediate(), idx, result_ty);
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
        if self.cx.tcx.is_thread_local_static(def_id) {
            // macOS thread-local access: load the variable's descriptor address, then call its
            // thunk (`descriptor[0]`), which returns the per-thread variable address in `x0`.
            let symref = SymRef { name: self.cx.sym_name(sym), addend: 0 };
            self.emit(Inst::AdrpTlv { rd: X0, sym: symref.clone() });
            self.emit(Inst::LdrTlvLo { rt: X0, rn: X0, sym: symref });
            self.emit(Inst::LoadStoreUImm {
                load: true,
                signed: false,
                size: MemSize::X,
                rt: Gpr::from_encoding(8),
                rn: X0,
                offset: 0,
            });
            self.emit(Inst::Blr { rn: Gpr::from_encoding(8) });
            let ptr_ty = self.ptr_ty();
            return self.spill(X0, ptr_ty);
        }
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
            // Determine the frame layout up front: scan the function's calls for the outgoing
            // stack-argument requirement so local slots can be placed above that area directly.
            let outgoing = match cx.cur_instance.get() {
                Some(instance) => outgoing_arg_bytes(cx, instance),
                None => 0,
            };
            let mut fb = FunctionBuild::new(llfn, name, is_global, outgoing as u64);
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
        } else if self.is_int128(v.ty()) {
            // A 128-bit integer is returned in the x0:x1 register pair.
            self.materialize128(v, X0, X1);
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
        if self.is_int128(lhs.ty()) {
            return self.int128_addsub(AddSub::Add, lhs, rhs, false);
        }
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
        if self.is_int128(lhs.ty()) {
            return self.int128_addsub(AddSub::Sub, lhs, rhs, false);
        }
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
        if self.is_int128(lhs.ty()) {
            return self.int128_mul(lhs, rhs);
        }
        self.alu_rrr(|size, rd, rn, rm| Inst::Madd { size, rd, rn, rm, ra: ZR }, lhs, rhs)
    }
    fn udiv(&mut self, lhs: Value, rhs: Value) -> Value {
        if self.is_int128(lhs.ty()) {
            return self.int128_bin_libcall("___udivti3", lhs, rhs);
        }
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
        if self.is_int128(lhs.ty()) {
            return self.int128_bin_libcall("___divti3", lhs, rhs);
        }
        self.alu_rrr_signed(
            |size, rd, rn, rm| Inst::DataProc2 { op: DataProc2::Sdiv, size, rd, rn, rm },
            lhs,
            rhs,
        )
    }
    fn exactsdiv(&mut self, lhs: Value, rhs: Value) -> Value {
        self.sdiv(lhs, rhs)
    }
    fn urem(&mut self, lhs: Value, rhs: Value) -> Value {
        if self.is_int128(lhs.ty()) {
            return self.int128_bin_libcall("___umodti3", lhs, rhs);
        }
        self.rem(lhs, rhs, DataProc2::Udiv)
    }
    fn srem(&mut self, lhs: Value, rhs: Value) -> Value {
        if self.is_int128(lhs.ty()) {
            return self.int128_bin_libcall("___modti3", lhs, rhs);
        }
        self.rem(lhs, rhs, DataProc2::Sdiv)
    }
    fn and(&mut self, lhs: Value, rhs: Value) -> Value {
        if self.is_int128(lhs.ty()) {
            return self.int128_logical(LogicOp::And, lhs, rhs);
        }
        self.alu_rrr(
            |size, rd, rn, rm| Inst::Logical { op: LogicOp::And, size, rd, rn, rm, amount: 0 },
            lhs,
            rhs,
        )
    }
    fn or(&mut self, lhs: Value, rhs: Value) -> Value {
        if self.is_int128(lhs.ty()) {
            return self.int128_logical(LogicOp::Orr, lhs, rhs);
        }
        self.alu_rrr(
            |size, rd, rn, rm| Inst::Logical { op: LogicOp::Orr, size, rd, rn, rm, amount: 0 },
            lhs,
            rhs,
        )
    }
    fn xor(&mut self, lhs: Value, rhs: Value) -> Value {
        if self.is_int128(lhs.ty()) {
            return self.int128_logical(LogicOp::Eor, lhs, rhs);
        }
        self.alu_rrr(
            |size, rd, rn, rm| Inst::Logical { op: LogicOp::Eor, size, rd, rn, rm, amount: 0 },
            lhs,
            rhs,
        )
    }
    fn shl(&mut self, lhs: Value, rhs: Value) -> Value {
        if self.is_int128(lhs.ty()) {
            return self.int128_shift("___ashlti3", lhs, rhs);
        }
        self.alu_rrr(
            |size, rd, rn, rm| Inst::DataProc2 { op: DataProc2::Lslv, size, rd, rn, rm },
            lhs,
            rhs,
        )
    }
    fn lshr(&mut self, lhs: Value, rhs: Value) -> Value {
        if self.is_int128(lhs.ty()) {
            return self.int128_shift("___lshrti3", lhs, rhs);
        }
        self.alu_rrr(
            |size, rd, rn, rm| Inst::DataProc2 { op: DataProc2::Lsrv, size, rd, rn, rm },
            lhs,
            rhs,
        )
    }
    fn ashr(&mut self, lhs: Value, rhs: Value) -> Value {
        if self.is_int128(lhs.ty()) {
            return self.int128_shift("___ashrti3", lhs, rhs);
        }
        // The shifted value must be sign-extended so the vacated high bits carry its sign; the
        // shift amount is a plain count and is loaded zero-extended.
        let ty = lhs.ty();
        let size = op_size(self.cx, ty);
        self.materialize_signed(lhs, X9);
        self.materialize(rhs, X10);
        self.emit(Inst::DataProc2 { op: DataProc2::Asrv, size, rd: X9, rn: X9, rm: X10 });
        self.spill(X9, ty)
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
        let n = match self.cx.type_data(val_ty) {
            TypeData::Int(b) => b,
            _ => 64,
        };
        let size = op_size(self.cx, lhs.ty());
        let op_bits = match size {
            OperandSize::S32 => 32,
            OperandSize::S64 => 64,
        };
        let bool_ty = self.cx.intern_type(TypeData::Int(1));
        // Sub-word add/sub: the NZCV flags reflect 32/64-bit overflow, not the narrow type's, so
        // `100i8 + 100i8` shows no flag overflow. Widen the operands (the op-width sum/difference
        // then cannot itself overflow), compute, and range-check against the type's bounds.
        if n < op_bits {
            let op_ty = self.cx.intern_type(TypeData::Int(op_bits));
            if signed {
                self.materialize_signed(lhs, X9);
                self.materialize_signed(rhs, X10);
            } else {
                self.materialize(lhs, X9);
                self.materialize(rhs, X10);
            }
            let addsub = match oop {
                OverflowOp::Add => AddSub::Add,
                OverflowOp::Sub => AddSub::Sub,
                OverflowOp::Mul => unreachable!(),
            };
            self.emit(Inst::AddSubReg {
                op: addsub,
                size,
                set_flags: false,
                rd: X9,
                rn: X9,
                rm: X10,
                amount: 0,
            });
            let wide = self.spill(X9, op_ty);
            let overflow = if signed {
                let max = self.cx.const_uint(op_ty, ((1i128 << (n - 1)) - 1) as u64);
                let min = self.cx.const_uint(op_ty, (-(1i128 << (n - 1))) as i64 as u64);
                let hi = self.icmp(IntPredicate::IntSGT, wide, max);
                let lo = self.icmp(IntPredicate::IntSLT, wide, min);
                self.or(hi, lo)
            } else {
                // Unsigned: a borrow on subtract wraps the op-width result above the max too.
                let max = self.cx.const_uint(op_ty, ((1u128 << n) - 1) as u64);
                self.icmp(IntPredicate::IntUGT, wide, max)
            };
            let result = self.trunc(wide, val_ty);
            return (result, overflow);
        }
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
        self.emit_frame_addr(X9, off);
        let ptr_ty = self.ptr_ty();
        self.spill(X9, ptr_ty)
    }
    fn alloca_with_ty(&mut self, layout: TyAndLayout<'tcx>) -> Value {
        self.alloca(layout.size, layout.align.abi)
    }

    fn load(&mut self, ty: Type, ptr: Value, _align: Align) -> Value {
        match self.cx.type_data(ty) {
            TypeData::Pair(..) => return self.load_pair(ty, ptr),
            TypeData::Vector(..) => {
                // A vector lives in a frame slot; load it by copying its bytes out of `[ptr]`.
                let (size, align) = self.cx.type_size_align(ty);
                let off = self.alloc_slot(size, align);
                self.materialize(ptr, X9);
                self.copy_ptr_to_slot(off, X9, size);
                return Value::Slot { off, ty };
            }
            TypeData::Int(b) if b > 64 => {
                // A 128-bit integer occupies 16 bytes; a single-register load would drop the high
                // word. Copy the whole value into a fresh slot.
                let (size, align) = self.cx.type_size_align(ty);
                let off = self.alloc_slot(size, align);
                self.materialize(ptr, X9);
                self.copy_ptr_to_slot(off, X9, size);
                return Value::Slot { off, ty };
            }
            _ => {}
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
        if let TypeData::Vector(..) = self.cx.type_data(val.ty()) {
            // Store a vector by copying its slot bytes into `[ptr]`.
            if let Some(off) = self.vector_to_slot(val) {
                let (size, _) = self.cx.type_size_align(val.ty());
                self.materialize(ptr, X9);
                self.copy_slot_to_ptr(off, X9, size);
            }
            return Value::Undef { ty: self.ptr_ty() };
        }
        if self.is_int128(val.ty()) {
            // A 128-bit integer is stored as its two 64-bit words.
            self.materialize(ptr, X9);
            self.materialize128(val, X10, X11);
            self.emit(Inst::LoadStoreUImm { load: false, signed: false, size: MemSize::X, rt: X10, rn: X9, offset: 0 });
            self.emit(Inst::LoadStoreUImm { load: false, signed: false, size: MemSize::X, rt: X11, rn: X9, offset: 8 });
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
        if self.is_int128(dest_ty) {
            // Sign-extend the source to 64 bits (the low word), then replicate its sign bit across
            // the high word (`asr #63`).
            let i64ty = self.cx.intern_type(TypeData::Int(64));
            let lo64 = self.sext(val, i64ty);
            self.materialize(lo64, X9);
            self.load_imm(X11, 63, OperandSize::S64);
            self.emit(Inst::DataProc2 { op: DataProc2::Asrv, size: OperandSize::S64, rd: X10, rn: X9, rm: X11 });
            return self.spill128(X9, X10, dest_ty);
        }
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
        if self.is_int128(dest_ty) {
            // Zero-extend the source into the low word; the high word is zero.
            self.materialize(val, X9);
            self.load_imm(X10, 0, OperandSize::S64);
            return self.spill128(X9, X10, dest_ty);
        }
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
        if self.is_int128(dest_ty) {
            return self.fp_to_int128(false, val, dest_ty);
        }
        let fp = fp_size(self.cx, val.ty());
        let int = op_size(self.cx, dest_ty);
        self.materialize_fp(val, V16);
        self.emit(Inst::FpToInt { signed: false, fp, int, rd: X9, rn: V16 });
        self.clamp_fp_to_int(X9, dest_ty, false)
    }
    fn fptosi(&mut self, val: Value, dest_ty: Type) -> Value {
        if self.is_int128(dest_ty) {
            return self.fp_to_int128(true, val, dest_ty);
        }
        let fp = fp_size(self.cx, val.ty());
        let int = op_size(self.cx, dest_ty);
        self.materialize_fp(val, V16);
        self.emit(Inst::FpToInt { signed: true, fp, int, rd: X9, rn: V16 });
        self.clamp_fp_to_int(X9, dest_ty, true)
    }
    fn uitofp(&mut self, val: Value, dest_ty: Type) -> Value {
        if self.is_int128(val.ty()) {
            return self.int128_to_fp(false, val, dest_ty);
        }
        // Zero-extend the source to 64 bits so any integer width converts correctly.
        let i64ty = self.cx.intern_type(TypeData::Int(64));
        let wide = self.zext(val, i64ty);
        let fp = fp_size(self.cx, dest_ty);
        self.materialize(wide, X9);
        self.emit(Inst::IntToFp { signed: false, fp, int: OperandSize::S64, rd: V16, rn: X9 });
        self.spill_fp(V16, dest_ty)
    }
    fn sitofp(&mut self, val: Value, dest_ty: Type) -> Value {
        if self.is_int128(val.ty()) {
            return self.int128_to_fp(true, val, dest_ty);
        }
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
        if self.is_int128(lhs.ty()) {
            return self.int128_icmp(op, lhs, rhs);
        }
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
        if self.is_int128(ty) {
            // Select each 64-bit word independently after a single `cmp cond, #0`.
            self.materialize(cond, X9);
            self.emit(Inst::AddSubImm { op: AddSub::Sub, size: OperandSize::S32, set_flags: true, rd: ZR, rn: X9, imm12: 0, shift12: false });
            self.materialize128(then_val, X10, X11);
            self.materialize128(else_val, X12, X13);
            self.emit(Inst::CondSel { op: CondSel::Csel, size: OperandSize::S64, rd: X10, rn: X10, rm: X12, cond: Cond::Ne });
            self.emit(Inst::CondSel { op: CondSel::Csel, size: OperandSize::S64, rd: X11, rn: X11, rm: X13, cond: Cond::Ne });
            return self.spill128(X10, X11, ty);
        }
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
            Value::Slot { off: base, .. } => Value::Slot { off: base + off as u64, ty: fty },
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
                self.alloc_slot(size, align)
            }
        };
        let field_off = base + foff as u64;
        if type_is_float(self.cx, elt.ty()) {
            self.materialize_fp(elt, V16);
            self.emit_mem_fp(false, fp_size(self.cx, elt.ty()), V16, SP, field_off);
        } else {
            self.materialize(elt, X9);
            self.emit_mem_gpr(false, false, mem_size(self.cx, elt.ty()), X9, SP, field_off);
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
        // Signed remainder needs sign-extended sub-word operands, exactly like signed division.
        if div == DataProc2::Sdiv {
            self.materialize_signed(lhs, X9);
            self.materialize_signed(rhs, X10);
        } else {
            self.materialize(lhs, X9);
            self.materialize(rhs, X10);
        }
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
        let base = self.alloc_slot(size, align);
        self.materialize(ptr, X9);
        for idx in 0..2 {
            let (foff, fty) = self.cx.pair_field(ty, idx);
            let foff = foff as u64;
            if type_is_float(self.cx, fty) {
                let sz = fp_size(self.cx, fty);
                self.emit(Inst::LoadStoreFpUImm { load: true, size: sz, rt: V16, rn: X9, offset: foff });
                self.emit_mem_fp(false, sz, V16, SP, base + foff);
            } else {
                let sz = mem_size(self.cx, fty);
                self.emit(Inst::LoadStoreUImm { load: true, signed: false, size: sz, rt: X10, rn: X9, offset: foff });
                self.emit_mem_gpr(false, false, sz, X10, SP, base + foff);
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
            let foff = foff as u64;
            if type_is_float(self.cx, fty) {
                let sz = fp_size(self.cx, fty);
                self.emit_mem_fp(true, sz, V16, SP, sbase + foff);
                self.emit(Inst::LoadStoreFpUImm { load: false, size: sz, rt: V16, rn: X9, offset: foff });
            } else {
                let sz = mem_size(self.cx, fty);
                self.emit_mem_gpr(true, false, sz, X10, SP, sbase + foff);
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
        // register banks; an sret pointer goes in x8. Once a bank is exhausted the remaining
        // arguments are written into the outgoing-argument area at the bottom of our frame
        // (`sp + nsaa`), which the frame-layout pre-pass sized so they never overlap local slots.
        let mut ngrn: u8 = 0;
        let mut nsrn: u8 = 0;
        let mut nsaa: u32 = 0;
        for (i, &arg) in args.iter().enumerate() {
            if indirect_ret && i == 0 {
                self.materialize(arg, Gpr::from_encoding(8));
            } else if type_is_float(self.cx, arg.ty()) {
                if nsrn < 8 {
                    self.materialize_fp(arg, Vreg::from_encoding(nsrn));
                    nsrn += 1;
                } else {
                    self.materialize_fp(arg, V16);
                    self.emit_mem_fp(false, fp_size(self.cx, arg.ty()), V16, SP, nsaa as u64);
                    nsaa += 8;
                }
            } else if self.is_int128(arg.ty()) {
                // A 128-bit integer takes two consecutive registers, else a 16-byte stack slot.
                if ngrn <= 6 {
                    let lo = Gpr::from_encoding(ngrn);
                    let hi = Gpr::from_encoding(ngrn + 1);
                    self.materialize128(arg, lo, hi);
                    ngrn += 2;
                } else {
                    ngrn = 8;
                    nsaa = (nsaa + 15) & !15;
                    self.materialize128(arg, X9, X10);
                    self.emit_mem_gpr(false, false, MemSize::X, X9, SP, nsaa as u64);
                    self.emit_mem_gpr(false, false, MemSize::X, X10, SP, nsaa as u64 + 8);
                    nsaa += 16;
                }
            } else if ngrn < 8 {
                self.materialize(arg, Gpr::from_encoding(ngrn));
                ngrn += 1;
            } else {
                self.materialize(arg, X9);
                self.emit_mem_gpr(false, false, mem_size(self.cx, arg.ty()), X9, SP, nsaa as u64);
                nsaa += 8;
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
                        let base = self.alloc_slot(size, align);
                        let mut ngrn: u8 = 0;
                        let mut nsrn: u8 = 0;
                        for (idx, fty) in [(0usize, fa), (1usize, fb)] {
                            let (foff, _) = self.cx.pair_field(ty, idx);
                            let field_off = base + foff as u64;
                            if type_is_float(self.cx, fty) {
                                self.emit_mem_fp(
                                    false,
                                    fp_size(self.cx, fty),
                                    Vreg::from_encoding(nsrn),
                                    SP,
                                    field_off,
                                );
                                nsrn += 1;
                            } else {
                                self.emit_mem_gpr(
                                    false,
                                    false,
                                    mem_size(self.cx, fty),
                                    Gpr::from_encoding(ngrn),
                                    SP,
                                    field_off,
                                );
                                ngrn += 1;
                            }
                        }
                        Value::Slot { off: base, ty }
                    }
                    _ if type_is_float(self.cx, ty) => self.spill_fp(V0, ty),
                    _ if self.is_int128(ty) => self.spill128(X0, X1, ty),
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
