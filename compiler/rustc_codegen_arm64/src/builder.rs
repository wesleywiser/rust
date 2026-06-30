//! The `Builder`: per-function machine-code construction.
//!
//! The baseline model keeps every SSA value in a stack slot (no register allocator). A `Builder`
//! appends [`Inst`]s to the current basic block of the [`FunctionBuild`] held by the context; when
//! the function is finished its blocks are concatenated (each prefixed with its label) into a
//! [`MachFunction`] and pushed to the module.

use std::ops::Deref;

use rustc_abi::{Align, BackendRepr, HasDataLayout, RegKind, Scalar, Size, TargetDataLayout, WrappingRange};
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
    FnAbiOf, FnAbiOfHelpers, HasTyCtxt, HasTypingEnv, LayoutOf, LayoutOfHelpers, TyAndLayout,
};
use rustc_middle::ty::{self, AtomicOrdering, Instance, Ty, TyCtxt};
use rustc_span::{Span, sym};
use rustc_target::callconv::{ArgAbi, CastTarget, FnAbi, PassMode};
use rustc_target::spec::{HasTargetSpec, Target};

use crate::context::{BasicBlock, CodegenCx, Function, Type, TypeData, Value};
use crate::mach::func::MachFunction;
use crate::mach::frame::FrameLayout;
use crate::mach::inst::{
    AddSub, AtomicRmwOp, CondSel, CryptoThreeOp, CryptoTwoOp, DataProc1, DataProc2, DmbOption, FpOp1,
    FpOp2, Inst, Label, LogicOp, MemSize, MovKind, PairIndex, SimdOp, SimdUnOp, SymRef,
};
use crate::mach::reg::{
    Cond, FpSize, Gpr, OperandSize, Vreg, FP, LR, SP, V0, V1, V16, V17, V18, X0, X1, X2, X3, X4, X9,
    X10, X11, X12, X13, X16, X17, ZR,
};

/// Lane-wise SIMD arithmetic operation kind for [`Builder::emit_simd_arith`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum SimdArith {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

/// Horizontal SIMD reduction kind for [`Builder::emit_simd_reduce`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum SimdReduce {
    Add,
    Mul,
    Max,
    Min,
    And,
    Or,
    Xor,
}

/// Lane-wise integer bit operation for [`Builder::emit_simd_bit_unary`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum SimdBitOp {
    Ctpop,
    Ctlz,
    Cttz,
    Bswap,
    Bitreverse,
}

/// State for the function currently being lowered.
pub struct FunctionBuild {
    pub name: Box<str>,
    pub is_global: bool,
    /// Instruction list for each basic block, indexed by block id (which is also its [`Label`]).
    pub blocks: Vec<Vec<Inst>>,
    pub frame: FrameLayout,
    /// Spilled location of each physical incoming parameter, indexed by physical param index.
    pub param_slots: Vec<Value>,
    /// EH call sites recorded by `invoke` (label ids); resolved to offsets at encode time.
    pub call_sites: Vec<crate::mach::func::MachCallSite>,
    /// Next synthetic label id for bracketing call sites (kept clear of block-id labels).
    pub next_cs_label: u32,
}

impl FunctionBuild {
    /// Create a function builder. `outgoing_bytes` is the size of the outgoing-argument area
    /// (computed up front by scanning the function's calls); it fixes where local slots begin.
    pub fn new(name: Box<str>, is_global: bool, outgoing_bytes: u64) -> FunctionBuild {
        FunctionBuild {
            name,
            is_global,
            blocks: Vec::new(),
            frame: FrameLayout::new(outgoing_bytes),
            param_slots: Vec::new(),
            call_sites: Vec::new(),
            next_cs_label: 0xF000_0000,
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
        let align = self.frame.alignment();
        // The stack pointer is only 16-byte aligned on entry. If any local needs more, the prologue
        // must realign `sp` down to `align` (and the epilogue restore it from the frame pointer,
        // since `sp` is then no longer a fixed distance below it).
        let realign = align > 16;
        let mut f = MachFunction::new(self.name, self.is_global);
        f.call_sites = self.call_sites;

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
        if realign {
            push_sp_realign(&mut f.insts, align);
        }

        for (id, block) in self.blocks.into_iter().enumerate() {
            f.push(Inst::Label(id as Label));
            for inst in block {
                if let Inst::Ret { .. } = inst {
                    // Epilogue before every return.
                    if realign {
                        // `sp` was realigned to an unknown distance below `fp`; restore it directly.
                        f.push(Inst::MovSp { size: OperandSize::S64, rd: SP, rn: FP });
                    } else {
                        push_sp_adjust(&mut f.insts, AddSub::Add, frame);
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
        FpSize::S16 => 2,
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

/// Push a 128-bit (`q`) FP load/store of `rt` at `[base, #offset]` (a whole `f128`), with the same
/// large-offset handling as [`push_mem_fp`].
fn push_mem_q(out: &mut Vec<Inst>, load: bool, rt: Vreg, base: Gpr, offset: u64) {
    if imm_offset_fits(offset, 16) {
        out.push(Inst::LoadStoreQ { load, rt, rn: base, offset });
    } else {
        push_load_imm64(out, X16, offset);
        out.push(Inst::AddSubExtReg { op: AddSub::Add, size: OperandSize::S64, rd: X16, rn: base, rm: X16 });
        out.push(Inst::LoadStoreQ { load, rt, rn: X16, offset: 0 });
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

/// Realign the stack pointer downward to `align` (a power of two greater than 16): `sp &= ~(align-1)`.
/// Computed via `x16`/`x17`, which are dead in the prologue (params are not spilled until the entry
/// block). The stack pointer can only be written through the SP-aware move/extended forms, so the
/// mask is applied in `x16` and moved back.
fn push_sp_realign(out: &mut Vec<Inst>, align: u64) {
    debug_assert!(align.is_power_of_two() && align > 16);
    let mask = !(align - 1);
    out.push(Inst::MovSp { size: OperandSize::S64, rd: X16, rn: SP });
    if align <= (1 << 16) {
        // `movn x17, #(align-1)` materializes `~(align-1)` = mask in one instruction.
        out.push(Inst::MovWide {
            kind: MovKind::Inverse,
            size: OperandSize::S64,
            rd: X17,
            imm16: (align - 1) as u16,
            shift: 0,
        });
    } else {
        push_load_imm64(out, X17, mask);
    }
    out.push(Inst::Logical {
        op: LogicOp::And,
        size: OperandSize::S64,
        rd: X16,
        rn: X16,
        rm: X17,
        amount: 0,
    });
    out.push(Inst::MovSp { size: OperandSize::S64, rd: SP, rn: X16 });
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

    /// Allocate a fresh synthetic label id for bracketing an EH call site (kept clear of block-id
    /// labels, which start at 0).
    fn fresh_cs_label(&self) -> u32 {
        let mut cur = self.cx.cur_fn.borrow_mut();
        let fb = cur.as_mut().expect("no function is currently being built");
        let l = fb.next_cs_label;
        fb.next_cs_label += 1;
        l
    }

    /// Record an EH call site: a call in `[begin, end)` that unwinds transfers to landing-pad
    /// block `lp` (`action` 0 = cleanup, 1 = catch-all).
    fn record_call_site(&self, begin: u32, end: u32, lp: u32, action: u8) {
        let mut cur = self.cx.cur_fn.borrow_mut();
        let fb = cur.as_mut().expect("no function is currently being built");
        fb.call_sites.push(crate::mach::func::MachCallSite { begin, end, landing_pad: lp, action });
    }

    /// Copy `size` bytes from `[src]` into the frame slot at `dst_off`, using descending
    /// power-of-two chunks through the `x10` scratch (`src` must not be `x10`/`x16`).
    fn copy_ptr_to_slot(&mut self, dst_off: u64, src: Gpr, size: u64) {
        let mut o = 0u64;
        for chunk in [8u64, 4, 2, 1] {
            let msize = mem_size_from_bytes(chunk);
            while o + chunk <= size {
                self.emit_mem_gpr(true, false, msize, X10, src, o);
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
                self.emit_mem_gpr(false, false, msize, X10, dst, o);
                o += chunk;
            }
        }
    }

    /// Copy `size` bytes from frame slot `src_off` to `[base + dst_off]`, using descending
    /// power-of-two chunks through the `x10` scratch. Used to assemble/scatter `PassMode::Cast`
    /// register pieces, whose offsets within an aggregate need not be 8-aligned. `base` and `dst`
    /// must not be `x10`.
    fn copy_slot_to_base_off(&mut self, src_off: u64, base: Gpr, dst_off: u64, size: u64) {
        let mut o = 0u64;
        for chunk in [8u64, 4, 2, 1] {
            let msize = mem_size_from_bytes(chunk);
            while o + chunk <= size {
                self.emit_mem_gpr(true, false, msize, X10, SP, src_off + o);
                self.emit_mem_gpr(false, false, msize, X10, base, dst_off + o);
                o += chunk;
            }
        }
    }

    /// Copy `size` bytes from `[base + src_off]` into frame slot `dst_off` (mirror of
    /// [`copy_slot_to_base_off`](Self::copy_slot_to_base_off)). `base` must not be `x10`.
    fn copy_base_off_to_slot(&mut self, base: Gpr, src_off: u64, dst_off: u64, size: u64) {
        let mut o = 0u64;
        for chunk in [8u64, 4, 2, 1] {
            let msize = mem_size_from_bytes(chunk);
            while o + chunk <= size {
                self.emit_mem_gpr(true, false, msize, X10, base, src_off + o);
                self.emit_mem_gpr(false, false, msize, X10, SP, dst_off + o);
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

    /// `simd_splat`: broadcast a scalar into every lane of a fresh vector slot. Integer lanes go
    /// through a GPR; floating-point lanes through an FP register.
    fn emit_simd_splat(&mut self, scalar: Value, vec_ty: Type) -> Value {
        let (elem, count, es) = self.vector_info(vec_ty);
        let (size, align) = self.cx.type_size_align(vec_ty);
        let off = self.alloc_slot(size, align);
        if type_is_float(self.cx, elem) {
            let fs = fp_size(self.cx, elem);
            self.materialize_fp(scalar, V16);
            for i in 0..count {
                self.emit_mem_fp(false, fs, V16, SP, off + i * es);
            }
            return Value::Slot { off, ty: vec_ty };
        }
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
        let is_float = type_is_float(self.cx, elem);
        // NEON fast path: native-width vectors map to a single `cm*`/`fcm*` instruction (with an
        // operand swap for the `<`/`<=` forms and a trailing `not` for `ne`). The mask lanes are the
        // same width as the input lanes, so the same `q`/size applies to the result store.
        if let Some((q, isize)) = self.simd_native(a.ty()) {
            if let Some((op, swap, negate)) = simd_cmp_neon(cond, is_float) {
                let size = if is_float { (es == 8) as u8 } else { isize };
                let (x, y) = if swap { (b, a) } else { (a, b) };
                let r = self.emit_simd_three(op, q, size, x, y, mask_ty);
                let r = if negate { self.emit_simd_two(SimdUnOp::Not, q, 0, r, mask_ty) } else { r };
                return r;
            }
        }
        let aoff = self.vector_to_slot(a);
        let boff = self.vector_to_slot(b);
        let (Some(aoff), Some(boff)) = (aoff, boff) else {
            return Value::Undef { ty: mask_ty };
        };
        let (melem, _mcount, mes) = self.vector_info(mask_ty);
        let (size, align) = self.cx.type_size_align(mask_ty);
        let roff = self.alloc_slot(size, align);
        let mmsize = mem_size(self.cx, melem);
        if type_is_float(self.cx, elem) {
            // Float lanes: fcmp + csetm. `lt`/`le` become `mi`/`ls` against the fcmp flags.
            let fcond = match cond {
                Cond::Lt | Cond::Lo => Cond::Mi,
                Cond::Le | Cond::Ls => Cond::Ls,
                Cond::Hi => Cond::Gt,
                Cond::Hs => Cond::Ge,
                other => other,
            };
            let fs = fp_size(self.cx, elem);
            for i in 0..count {
                self.emit_mem_fp(true, fs, V16, SP, aoff + i * es);
                self.emit_mem_fp(true, fs, V17, SP, boff + i * es);
                self.emit(Inst::FpCmp { size: fs, rn: V16, rm: V17 });
                self.emit(Inst::CondSel { op: CondSel::Csinv, size: OperandSize::S64, rd: X10, rn: ZR, rm: ZR, cond: fcond.invert() });
                self.emit_mem_gpr(false, false, mmsize, X10, SP, roff + i * mes);
            }
            return Value::Slot { off: roff, ty: mask_ty };
        }
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

    /// `simd_and`/`simd_or`/`simd_xor`: lane-wise bitwise operation. Native-width vectors use a
    /// single NEON `and`/`orr`/`eor`; wider vectors fall through to the per-lane path. Integer lanes
    /// only.
    fn emit_simd_binop(&mut self, op: LogicOp, a: Value, b: Value) -> Value {
        let vec_ty = a.ty();
        let (elem, count, es) = self.vector_info(vec_ty);
        if let Some((q, _)) = self.simd_native(vec_ty) {
            let sop = match op {
                LogicOp::And => Some(SimdOp::And),
                LogicOp::Orr => Some(SimdOp::Orr),
                LogicOp::Eor => Some(SimdOp::Eor),
                _ => None,
            };
            if let Some(sop) = sop {
                return self.emit_simd_three(sop, q, 0, a, b, vec_ty);
            }
        }
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

    /// The NEON arrangement `(q, size)` for a vector that fits a single `v` register — exactly 8
    /// (`q = false`) or 16 (`q = true`) bytes — or `None` for wider vectors (handled by the scalar
    /// per-lane path). `size` is the integer lane-size code (`0`/`1`/`2`/`3` = 8/16/32/64-bit); the
    /// float ops re-derive their single `sz` bit from the element size.
    fn simd_native(&self, vec_ty: Type) -> Option<(bool, u8)> {
        let (_elem, _count, es) = self.vector_info(vec_ty);
        let (vbytes, _) = self.cx.type_size_align(vec_ty);
        let q = match vbytes {
            16 => true,
            8 => false,
            _ => return None,
        };
        let size = match es {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            _ => return None,
        };
        Some((q, size))
    }

    /// Load (`load`) or store a native-width vector (`q` ? 16 : 8 bytes) between frame offset `off`
    /// and the `v` register `vreg`.
    fn emit_vec_mem(&mut self, load: bool, q: bool, vreg: Vreg, off: u64) {
        if q {
            self.emit_q(load, vreg, off);
        } else {
            self.emit_mem_fp(load, FpSize::S64, vreg, SP, off);
        }
    }

    /// Emit a NEON 3-same vector op (`rd = a <op> b`) on native-width vectors, moving the operands
    /// through `v` registers. `size` is the encoding's size/`sz` field for `op`.
    fn emit_simd_three(
        &mut self,
        op: SimdOp,
        q: bool,
        size: u8,
        a: Value,
        b: Value,
        vec_ty: Type,
    ) -> Value {
        let aoff = self.vector_to_slot(a);
        let boff = self.vector_to_slot(b);
        let (Some(aoff), Some(boff)) = (aoff, boff) else {
            return Value::Undef { ty: vec_ty };
        };
        self.emit_vec_mem(true, q, V16, aoff);
        self.emit_vec_mem(true, q, V17, boff);
        self.emit(Inst::SimdThree { op, q, size, rd: V16, rn: V16, rm: V17 });
        let (vsize, valign) = self.cx.type_size_align(vec_ty);
        let roff = self.alloc_slot(vsize, valign);
        self.emit_vec_mem(false, q, V16, roff);
        Value::Slot { off: roff, ty: vec_ty }
    }

    /// Emit a NEON 2-register-misc op (`rd = <op> a`) on a native-width vector, moving the operand
    /// through a `v` register. `size` is the encoding's size/`sz` field for `op`.
    fn emit_simd_two(&mut self, op: SimdUnOp, q: bool, size: u8, a: Value, vec_ty: Type) -> Value {
        let Some(aoff) = self.vector_to_slot(a) else {
            return Value::Undef { ty: vec_ty };
        };
        self.emit_vec_mem(true, q, V16, aoff);
        self.emit(Inst::SimdTwo { op, q, size, rd: V16, rn: V16 });
        let (vsize, valign) = self.cx.type_size_align(vec_ty);
        let roff = self.alloc_slot(vsize, valign);
        self.emit_vec_mem(false, q, V16, roff);
        Value::Slot { off: roff, ty: vec_ty }
    }

    /// Lane-wise SIMD arithmetic (`simd_add`/`simd_sub`/`simd_mul`/`simd_div`/`simd_rem`).
    /// Native-width (8/16-byte) vectors use a single NEON instruction (`add`/`sub`/`mul`,
    /// `fadd`/`fsub`/`fmul`/`fdiv`) over `v` registers. The cases without a NEON instruction —
    /// integer `div`/`rem`, 64-bit-lane integer `mul`, and float `rem` — plus any wider-than-128-bit
    /// vector fall through to the per-lane path below: integer lanes use the GPR divide unit
    /// (`udiv`/`sdiv` + `msub`), float `rem` calls `fmod`/`fmodf` per lane. `signed` selects
    /// `sdiv`/`srem` (sign-extending sub-word lanes first, since loads zero-extend).
    fn emit_simd_arith(&mut self, a: Value, b: Value, kind: SimdArith, signed: bool) -> Value {
        let vec_ty = a.ty();
        let (elem, count, es) = self.vector_info(vec_ty);
        let is_float = type_is_float(self.cx, elem);
        // NEON fast path for native-width vectors with a corresponding vector instruction.
        if let Some((q, isize)) = self.simd_native(vec_ty) {
            let fsz = (es == 8) as u8; // float sz bit: 0 = f32, 1 = f64
            let neon = match (kind, is_float) {
                (SimdArith::Add, false) => Some((SimdOp::Add, isize)),
                (SimdArith::Sub, false) => Some((SimdOp::Sub, isize)),
                // NEON has no 64-bit-lane integer multiply.
                (SimdArith::Mul, false) if es != 8 => Some((SimdOp::Mul, isize)),
                (SimdArith::Add, true) => Some((SimdOp::Fadd, fsz)),
                (SimdArith::Sub, true) => Some((SimdOp::Fsub, fsz)),
                (SimdArith::Mul, true) => Some((SimdOp::Fmul, fsz)),
                (SimdArith::Div, true) => Some((SimdOp::Fdiv, fsz)),
                _ => None,
            };
            if let Some((op, size)) = neon {
                return self.emit_simd_three(op, q, size, a, b, vec_ty);
            }
        }
        let aoff = self.vector_to_slot(a);
        let boff = self.vector_to_slot(b);
        let (Some(aoff), Some(boff)) = (aoff, boff) else {
            return Value::Undef { ty: vec_ty };
        };
        let (size, align) = self.cx.type_size_align(vec_ty);
        let roff = self.alloc_slot(size, align);
        if is_float {
            let fs = fp_size(self.cx, elem);
            // `rem` is `fmod`/`fmodf` per lane (no FP remainder instruction).
            if kind == SimdArith::Rem {
                let sym = self.libm_symbol("fmod", elem);
                for i in 0..count {
                    self.emit_mem_fp(true, fs, V0, SP, aoff + i * es);
                    self.emit_mem_fp(true, fs, V1, SP, boff + i * es);
                    self.emit(Inst::Bl { sym: SymRef::new(sym.clone()) });
                    self.emit_mem_fp(false, fs, V0, SP, roff + i * es);
                }
                return Value::Slot { off: roff, ty: vec_ty };
            }
            let op = match kind {
                SimdArith::Add => crate::mach::inst::FpOp2::Fadd,
                SimdArith::Sub => crate::mach::inst::FpOp2::Fsub,
                SimdArith::Mul => crate::mach::inst::FpOp2::Fmul,
                _ => crate::mach::inst::FpOp2::Fdiv,
            };
            for i in 0..count {
                self.emit_mem_fp(true, fs, V16, SP, aoff + i * es);
                self.emit_mem_fp(true, fs, V17, SP, boff + i * es);
                self.emit(Inst::FpDataProc2 { op, size: fs, rd: V16, rn: V16, rm: V17 });
                self.emit_mem_fp(false, fs, V16, SP, roff + i * es);
            }
            return Value::Slot { off: roff, ty: vec_ty };
        }
        let msize = mem_size(self.cx, elem);
        for i in 0..count {
            self.emit_mem_gpr(true, false, msize, X10, SP, aoff + i * es);
            self.emit_mem_gpr(true, false, msize, X11_HACK, SP, boff + i * es);
            match kind {
                SimdArith::Add => self.emit(Inst::AddSubReg { op: AddSub::Add, size: OperandSize::S64, set_flags: false, rd: X10, rn: X10, rm: X11_HACK, amount: 0 }),
                SimdArith::Sub => self.emit(Inst::AddSubReg { op: AddSub::Sub, size: OperandSize::S64, set_flags: false, rd: X10, rn: X10, rm: X11_HACK, amount: 0 }),
                SimdArith::Mul => self.emit(Inst::Madd { size: OperandSize::S64, rd: X10, rn: X10, rm: X11_HACK, ra: ZR }),
                SimdArith::Div | SimdArith::Rem => {
                    // NEON has no integer divide; do it per lane in a GPR. Signed lanes must be
                    // sign-extended first (loads zero-extend); the quotient/remainder of two
                    // in-range lane values stays in range, so the low bytes are stored back.
                    if signed {
                        self.sign_extend_reg(X10, es);
                        self.sign_extend_reg(X11_HACK, es);
                    }
                    let divop = if signed { DataProc2::Sdiv } else { DataProc2::Udiv };
                    if kind == SimdArith::Div {
                        self.emit(Inst::DataProc2 { op: divop, size: OperandSize::S64, rd: X10, rn: X10, rm: X11_HACK });
                    } else {
                        // rem = a - (a / b) * b
                        self.emit(Inst::DataProc2 { op: divop, size: OperandSize::S64, rd: X12, rn: X10, rm: X11_HACK });
                        self.emit(Inst::Msub { size: OperandSize::S64, rd: X10, rn: X12, rm: X11_HACK, ra: X10 });
                    }
                }
            }
            self.emit_mem_gpr(false, false, msize, X10, SP, roff + i * es);
        }
        Value::Slot { off: roff, ty: vec_ty }
    }

    /// `simd_select(mask, if_true, if_false)`: per-lane blend. Each mask lane is all-ones (true) or
    /// all-zero (false); the lane value is `if_true` when the mask is non-zero, else `if_false`.
    /// Values are copied through a GPR (bit-preserving, so float lanes work too).
    fn emit_simd_select(&mut self, mask: Value, a: Value, b: Value, result_ty: Type) -> Value {
        let (melem, count, mes) = self.vector_info(mask.ty());
        let (velem, _vcount, ves) = self.vector_info(result_ty);
        let moff = self.vector_to_slot(mask);
        let aoff = self.vector_to_slot(a);
        let boff = self.vector_to_slot(b);
        let (Some(moff), Some(aoff), Some(boff)) = (moff, aoff, boff) else {
            return Value::Undef { ty: result_ty };
        };
        let (size, align) = self.cx.type_size_align(result_ty);
        let roff = self.alloc_slot(size, align);
        let mmsize = mem_size(self.cx, melem);
        let vmsize = mem_size(self.cx, velem);
        for i in 0..count {
            self.emit_mem_gpr(true, false, mmsize, X10, SP, moff + i * mes);
            self.emit_mem_gpr(true, false, vmsize, X11_HACK, SP, aoff + i * ves);
            self.emit_mem_gpr(true, false, vmsize, X12, SP, boff + i * ves);
            // result = (mask != 0) ? a : b
            self.emit(Inst::AddSubImm {
                op: AddSub::Sub,
                size: OperandSize::S64,
                set_flags: true,
                rd: ZR,
                rn: X10,
                imm12: 0,
                shift12: false,
            });
            self.emit(Inst::CondSel {
                op: CondSel::Csel,
                size: OperandSize::S64,
                rd: X11_HACK,
                rn: X11_HACK,
                rm: X12,
                cond: Cond::Ne,
            });
            self.emit_mem_gpr(false, false, vmsize, X11_HACK, SP, roff + i * ves);
        }
        Value::Slot { off: roff, ty: result_ty }
    }

    /// Lane-wise SIMD floating-point unary op (`simd_fabs`/`fsqrt`/`ceil`/`floor`/`round`/`trunc`).
    fn emit_simd_fp_unary(&mut self, a: Value, op: crate::mach::inst::FpOp1) -> Value {
        let vec_ty = a.ty();
        let (elem, count, es) = self.vector_info(vec_ty);
        // NEON fast path: native-width float vectors map to a single two-register-misc instruction.
        if let Some((q, _)) = self.simd_native(vec_ty) {
            if let Some(unop) = fp_op1_to_simd(op) {
                let sz = (es == 8) as u8; // float sz bit: 0 = f32, 1 = f64
                return self.emit_simd_two(unop, q, sz, a, vec_ty);
            }
        }
        let Some(aoff) = self.vector_to_slot(a) else { return Value::Undef { ty: vec_ty } };
        let (size, align) = self.cx.type_size_align(vec_ty);
        let roff = self.alloc_slot(size, align);
        let fs = fp_size(self.cx, elem);
        for i in 0..count {
            self.emit_mem_fp(true, fs, V16, SP, aoff + i * es);
            self.emit(Inst::FpDataProc1 { op, size: fs, rd: V16, rn: V16 });
            self.emit_mem_fp(false, fs, V16, SP, roff + i * es);
        }
        Value::Slot { off: roff, ty: vec_ty }
    }

    /// `simd_fma`: lane-wise fused multiply-add `a*b + c`.
    fn emit_simd_fma(&mut self, a: Value, b: Value, c: Value) -> Value {
        let vec_ty = a.ty();
        let (elem, count, es) = self.vector_info(vec_ty);
        let (Some(aoff), Some(boff), Some(coff)) = (self.vector_to_slot(a), self.vector_to_slot(b), self.vector_to_slot(c)) else {
            return Value::Undef { ty: vec_ty };
        };
        let (size, align) = self.cx.type_size_align(vec_ty);
        let roff = self.alloc_slot(size, align);
        let fs = fp_size(self.cx, elem);
        for i in 0..count {
            self.emit_mem_fp(true, fs, V16, SP, aoff + i * es);
            self.emit_mem_fp(true, fs, V17, SP, boff + i * es);
            self.emit_mem_fp(true, fs, V0, SP, coff + i * es);
            self.emit(Inst::FpFma { size: fs, rd: V16, rn: V16, rm: V17, ra: V0 });
            self.emit_mem_fp(false, fs, V16, SP, roff + i * es);
        }
        Value::Slot { off: roff, ty: vec_ty }
    }

    /// `simd_shl`/`simd_shr`: lane-wise variable shift (each lane of `a` shifted by the
    /// corresponding lane of `b`). For a right shift, `arith` selects arithmetic (sign-propagating)
    /// vs logical; the lane is sign-extended first for the arithmetic case so the sign fills from
    /// the lane's own top bit rather than the zero-extended register's. Integer lanes only.
    fn emit_simd_shift(&mut self, a: Value, b: Value, left: bool, arith: bool) -> Value {
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
        let op = if left {
            DataProc2::Lslv
        } else if arith {
            DataProc2::Asrv
        } else {
            DataProc2::Lsrv
        };
        for i in 0..count {
            self.emit_mem_gpr(true, false, msize, X10, SP, aoff + i * es);
            if !left && arith {
                self.sign_extend_reg(X10, es);
            }
            self.emit_mem_gpr(true, false, msize, X11_HACK, SP, boff + i * es);
            self.emit(Inst::DataProc2 { op, size: OperandSize::S64, rd: X10, rn: X10, rm: X11_HACK });
            self.emit_mem_gpr(false, false, msize, X10, SP, roff + i * es);
        }
        Value::Slot { off: roff, ty: vec_ty }
    }

    /// `simd_cast`/`simd_as`: lane-wise numeric cast (`as` per lane), with the lane count unchanged.
    /// Integer↔integer narrows by truncation and widens by sign/zero extension (per the source
    /// signedness). Casts involving floating-point lanes (int↔float, float↔float) reuse the scalar
    /// conversion lowering per lane, which saturates float→int and maps NaN to zero (matching `as`).
    fn emit_simd_cast(
        &mut self,
        src: Value,
        src_signed: bool,
        dst_signed: bool,
        dst_ty: Type,
    ) -> Value {
        let (src_elem, count, src_es) = self.vector_info(src.ty());
        let (dst_elem, _dcount, dst_es) = self.vector_info(dst_ty);
        let src_float = type_is_float(self.cx, src_elem);
        let dst_float = type_is_float(self.cx, dst_elem);
        let Some(soff) = self.vector_to_slot(src) else {
            return Value::Undef { ty: dst_ty };
        };
        let (size, align) = self.cx.type_size_align(dst_ty);
        let roff = self.alloc_slot(size, align);

        if !src_float && !dst_float {
            // Integer→integer fast path: load each lane, sign-extend a widening signed source, and
            // store the destination-width low bytes.
            let src_msize = mem_size(self.cx, src_elem);
            let dst_msize = mem_size(self.cx, dst_elem);
            for i in 0..count {
                self.emit_mem_gpr(true, false, src_msize, X10, SP, soff + i * src_es);
                if src_signed && dst_es > src_es {
                    self.sign_extend_reg(X10, src_es);
                }
                self.emit_mem_gpr(false, false, dst_msize, X10, SP, roff + i * dst_es);
            }
            return Value::Slot { off: roff, ty: dst_ty };
        }

        // At least one side is floating-point: convert each lane through the scalar cast helpers
        // (which materialize the lane in a register, convert, and clamp as needed).
        let dst_msize = mem_size(self.cx, dst_elem);
        let dst_fp = if dst_float { Some(fp_size(self.cx, dst_elem)) } else { None };
        for i in 0..count {
            let lane = Value::Slot { off: soff + i * src_es, ty: src_elem };
            let converted = match (src_float, dst_float) {
                (false, true) => {
                    if src_signed {
                        self.sitofp(lane, dst_elem)
                    } else {
                        self.uitofp(lane, dst_elem)
                    }
                }
                (true, false) => {
                    if dst_signed {
                        self.fptosi(lane, dst_elem)
                    } else {
                        self.fptoui(lane, dst_elem)
                    }
                }
                // float→float: widen with `fpext`, narrow with `fptrunc` (both lower to `fcvt`).
                (true, true) => {
                    if dst_es > src_es {
                        self.fpext(lane, dst_elem)
                    } else {
                        self.fptrunc(lane, dst_elem)
                    }
                }
                (false, false) => unreachable!(),
            };
            if let Some(fp) = dst_fp {
                self.materialize_fp(converted, V16);
                self.emit_mem_fp(false, fp, V16, SP, roff + i * dst_es);
            } else {
                self.materialize(converted, X10);
                self.emit_mem_gpr(false, false, dst_msize, X10, SP, roff + i * dst_es);
            }
        }
        Value::Slot { off: roff, ty: dst_ty }
    }

    /// `llvm.aarch64.neon.umaxp` (`vpmaxq_u8` etc.): pairwise unsigned maximum. For inputs `a` and
    /// `b` of `N` lanes, the result's first `N/2` lanes are the maxima of adjacent pairs of `a` and
    /// the second `N/2` lanes the maxima of adjacent pairs of `b`. Integer lanes only.
    fn emit_neon_umaxp(&mut self, a: Value, b: Value) -> Value {
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
        let half = count / 2;
        for (src_off, base) in [(aoff, 0u64), (boff, half)] {
            for j in 0..half {
                self.emit_mem_gpr(true, false, msize, X10, SP, src_off + (2 * j) * es);
                self.emit_mem_gpr(true, false, msize, X11_HACK, SP, src_off + (2 * j + 1) * es);
                // x10 = max(x10, x11): keep x10 when it is the (unsigned) greater, else x11.
                self.emit(Inst::AddSubReg { op: AddSub::Sub, size: OperandSize::S64, set_flags: true, rd: ZR, rn: X10, rm: X11_HACK, amount: 0 });
                self.emit(Inst::CondSel { op: CondSel::Csel, size: OperandSize::S64, rd: X10, rn: X10, rm: X11_HACK, cond: Cond::Hs });
                self.emit_mem_gpr(false, false, msize, X10, SP, roff + (base + j) * es);
            }
        }
        Value::Slot { off: roff, ty: vec_ty }
    }

    /// `llvm.aarch64.neon.tbl1` (`vqtbl1q_u8`, used by aho-corasick's Teddy matcher): a byte table
    /// lookup. For each index lane `i`, `result[i] = table[index[i]]` when `index[i]` is in range,
    /// otherwise `0` (per the LLVM LangRef / NEON `TBL`). The table lives in a frame slot, so the
    /// per-lane read is a register-indexed byte load; the index is clamped to keep the load
    /// in-bounds and the result masked to zero when the original index was out of range.
    fn emit_neon_tbl1(&mut self, table: Value, indices: Value) -> Value {
        let result_ty = indices.ty();
        let (_telem, tcount, _tes) = self.vector_info(table.ty());
        let (ielem, icount, ies) = self.vector_info(result_ty);
        let toff = self.vector_to_slot(table);
        let ioff = self.vector_to_slot(indices);
        let (Some(toff), Some(ioff)) = (toff, ioff) else {
            return Value::Undef { ty: result_ty };
        };
        let (size, align) = self.cx.type_size_align(result_ty);
        let roff = self.alloc_slot(size, align);
        let imsize = mem_size(self.cx, ielem);
        debug_assert!(tcount <= 0xfff, "tbl1 table lane count exceeds 12-bit bounds immediate");
        for i in 0..icount {
            self.emit_mem_gpr(true, false, imsize, X10, SP, ioff + i * ies);
            // cmp index, #tcount  -> sets the carry used by both selects below.
            self.emit(Inst::AddSubImm { op: AddSub::Sub, size: OperandSize::S64, set_flags: true, rd: ZR, rn: X10, imm12: tcount as u16, shift12: false });
            // Safe load index (the original index when in range, else 0) and an all-ones/zero mask.
            self.emit(Inst::CondSel { op: CondSel::Csel, size: OperandSize::S64, rd: X12, rn: X10, rm: ZR, cond: Cond::Lo });
            self.emit(Inst::CondSel { op: CondSel::Csinv, size: OperandSize::S64, rd: X9, rn: ZR, rm: ZR, cond: Cond::Lo.invert() });
            self.emit_frame_addr(X13, toff);
            self.emit(Inst::AddSubReg { op: AddSub::Add, size: OperandSize::S64, set_flags: false, rd: X13, rn: X13, rm: X12, amount: 0 });
            self.emit(Inst::LoadStoreUImm { load: true, signed: false, size: MemSize::B, rt: X11_HACK, rn: X13, offset: 0 });
            self.emit(Inst::Logical { op: LogicOp::And, size: OperandSize::S64, rd: X11_HACK, rn: X11_HACK, rm: X9, amount: 0 });
            self.emit_mem_gpr(false, false, imsize, X11_HACK, SP, roff + i * ies);
        }
        Value::Slot { off: roff, ty: result_ty }
    }

    /// The backend type of an intrinsic call's return value, derived from the monomorphized
    /// instance signature. Used by the NEON intrinsics whose result vector type differs from their
    /// argument types (the widening / pairwise-long ops).
    fn intrinsic_result_ty(&self, instance: Instance<'tcx>) -> Type {
        let tcx = self.cx.tcx;
        let fn_ty = instance.ty(tcx, ty::TypingEnv::fully_monomorphized());
        let sig = fn_ty.fn_sig(tcx);
        let ret = tcx.instantiate_bound_regions_with_erased(sig.output());
        self.cx.immediate_backend_type(self.cx.layout_of(ret))
    }

    /// `llvm.aarch64.neon.uaddlp`/`saddlp` (`vpaddlq_*`, `vpadalq_*`): pairwise add long. Sums each
    /// adjacent pair of input lanes and widens the result to the next-larger lane type, so the
    /// output has half as many lanes. Signed inputs are sign-extended before the add (loads
    /// zero-extend). Integer lanes only.
    fn emit_neon_addlp(&mut self, a: Value, signed: bool, ret_ty: Type) -> Value {
        let (ielem, _icount, ies) = self.vector_info(a.ty());
        let Some(aoff) = self.vector_to_slot(a) else {
            return Value::Undef { ty: ret_ty };
        };
        let (oelem, ocount, oes) = self.vector_info(ret_ty);
        let (size, align) = self.cx.type_size_align(ret_ty);
        let roff = self.alloc_slot(size, align);
        let imsize = mem_size(self.cx, ielem);
        let omsize = mem_size(self.cx, oelem);
        for j in 0..ocount {
            self.emit_mem_gpr(true, false, imsize, X10, SP, aoff + (2 * j) * ies);
            self.emit_mem_gpr(true, false, imsize, X11_HACK, SP, aoff + (2 * j + 1) * ies);
            if signed {
                self.sign_extend_reg(X10, ies);
                self.sign_extend_reg(X11_HACK, ies);
            }
            self.emit(Inst::AddSubReg { op: AddSub::Add, size: OperandSize::S64, set_flags: false, rd: X10, rn: X10, rm: X11_HACK, amount: 0 });
            self.emit_mem_gpr(false, false, omsize, X10, SP, roff + j * oes);
        }
        Value::Slot { off: roff, ty: ret_ty }
    }

    /// `llvm.aarch64.neon.addp` (`vpadd_*`): pairwise add. The result's lower half holds the
    /// adjacent-pair sums of `a`, the upper half those of `b` (same lane width as the inputs).
    /// Integer lanes only.
    fn emit_neon_addp(&mut self, a: Value, b: Value, ret_ty: Type) -> Value {
        let (elem, count, es) = self.vector_info(ret_ty);
        let aoff = self.vector_to_slot(a);
        let boff = self.vector_to_slot(b);
        let (Some(aoff), Some(boff)) = (aoff, boff) else {
            return Value::Undef { ty: ret_ty };
        };
        let (size, align) = self.cx.type_size_align(ret_ty);
        let roff = self.alloc_slot(size, align);
        let msize = mem_size(self.cx, elem);
        let half = count / 2;
        for (src_off, base) in [(aoff, 0u64), (boff, half)] {
            for j in 0..half {
                self.emit_mem_gpr(true, false, msize, X10, SP, src_off + (2 * j) * es);
                self.emit_mem_gpr(true, false, msize, X11_HACK, SP, src_off + (2 * j + 1) * es);
                self.emit(Inst::AddSubReg { op: AddSub::Add, size: OperandSize::S64, set_flags: false, rd: X10, rn: X10, rm: X11_HACK, amount: 0 });
                self.emit_mem_gpr(false, false, msize, X10, SP, roff + (base + j) * es);
            }
        }
        Value::Slot { off: roff, ty: ret_ty }
    }

    /// `llvm.aarch64.neon.umull`/`smull` (`vmull_*`): widening multiply. Each lane's product is
    /// computed at double width, so the output lane type is the next size up (the lane count is
    /// unchanged). Signed inputs are sign-extended first. Integer lanes only.
    fn emit_neon_mull(&mut self, a: Value, b: Value, signed: bool, ret_ty: Type) -> Value {
        let (ielem, icount, ies) = self.vector_info(a.ty());
        let aoff = self.vector_to_slot(a);
        let boff = self.vector_to_slot(b);
        let (Some(aoff), Some(boff)) = (aoff, boff) else {
            return Value::Undef { ty: ret_ty };
        };
        let (oelem, _ocount, oes) = self.vector_info(ret_ty);
        let (size, align) = self.cx.type_size_align(ret_ty);
        let roff = self.alloc_slot(size, align);
        let imsize = mem_size(self.cx, ielem);
        let omsize = mem_size(self.cx, oelem);
        for i in 0..icount {
            self.emit_mem_gpr(true, false, imsize, X10, SP, aoff + i * ies);
            self.emit_mem_gpr(true, false, imsize, X11_HACK, SP, boff + i * ies);
            if signed {
                self.sign_extend_reg(X10, ies);
                self.sign_extend_reg(X11_HACK, ies);
            }
            self.emit(Inst::Madd { size: OperandSize::S64, rd: X10, rn: X10, rm: X11_HACK, ra: ZR });
            self.emit_mem_gpr(false, false, omsize, X10, SP, roff + i * oes);
        }
        Value::Slot { off: roff, ty: ret_ty }
    }

    /// One AArch64 CRC32 step (`crc32{c}{b,h,w,x}`): combine the 32-bit accumulator `acc` with the
    /// data operand `data`, producing the next accumulator. `size` is the data-operand width
    /// (`S64` only for the `x`/`cx` forms); the accumulator is always 32-bit.
    fn emit_crc32(&mut self, op: DataProc2, size: OperandSize, acc: Value, data: Value) -> Value {
        let ret_ty = acc.ty();
        self.materialize(acc, X10);
        self.materialize(data, X11_HACK);
        self.emit(Inst::DataProc2 { op, size, rd: X10, rn: X10, rm: X11_HACK });
        self.spill(X10, ret_ty)
    }

    /// Emit a 128-bit (`q`) load/store of `vreg` at frame offset `off`, forming the address in the
    /// dedicated scratch register `X16` when the offset is too large for the scaled immediate.
    fn emit_q(&mut self, load: bool, vreg: Vreg, off: u64) {
        if off % 16 == 0 && off / 16 < (1 << 12) {
            self.emit(Inst::LoadStoreQ { load, rt: vreg, rn: SP, offset: off });
        } else {
            self.emit_frame_addr(X16, off);
            self.emit(Inst::LoadStoreQ { load, rt: vreg, rn: X16, offset: 0 });
        }
    }

    /// Load a 128-bit vector value (from its frame slot, materializing a constant vector first) into
    /// the `q` register `vreg`.
    fn load_q(&mut self, val: Value, vreg: Vreg) {
        if let Some(off) = self.vector_to_slot(val) {
            self.emit_q(true, vreg, off);
        }
    }

    /// Store the `q` register `vreg` to a fresh 16-byte slot, typed `ty`.
    fn store_q(&mut self, vreg: Vreg, ty: Type) -> Value {
        let off = self.alloc_slot(16, 16);
        self.emit_q(false, vreg, off);
        Value::Slot { off, ty }
    }

    /// Load an `i32` scalar into the low word of the `s` register `vreg` (SHA-1's `hash_e` operand).
    fn load_s32(&mut self, val: Value, vreg: Vreg) {
        self.materialize(val, X10);
        self.emit(Inst::FmovFromGpr { size: FpSize::S32, rd: vreg, rn: X10 });
    }

    /// Store the low word of the `s` register `vreg` to a fresh slot, typed `ty` (`sha1h`'s result).
    fn store_s32(&mut self, vreg: Vreg, ty: Type) -> Value {
        let off = self.alloc_slot(4, 4);
        self.emit_mem_fp(false, FpSize::S32, vreg, SP, off);
        Value::Slot { off, ty }
    }


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

    /// Horizontal reduction of a vector to a scalar (`simd_reduce_*`). `add`/`mul` are the *ordered*
    /// forms and take an initial accumulator (`init`); the others fold from lane 0. Integer `max`/
    /// `min` use `signed` to pick the comparison; `add`/`mul`/bitwise ops fold in 64-bit and
    /// truncate to the lane width on store. Floating-point `add`/`mul` fold left-to-right with the
    /// matching FP instruction (preserving the ordered semantics).
    fn emit_simd_reduce(
        &mut self,
        op: SimdReduce,
        x: Value,
        init: Option<Value>,
        signed: bool,
        result_ty: Type,
    ) -> Value {
        let (elem, count, es) = self.vector_info(x.ty());
        let Some(off) = self.vector_to_slot(x) else {
            return Value::Undef { ty: result_ty };
        };

        if type_is_float(self.cx, elem) {
            let fp = fp_size(self.cx, elem);
            let fpop = match op {
                SimdReduce::Add => FpOp2::Fadd,
                SimdReduce::Mul => FpOp2::Fmul,
                _ => self.cx.tcx.dcx().fatal(
                    "rustc_codegen_arm64: floating-point SIMD min/max/bitwise reductions are not \
                     yet supported",
                ),
            };
            // Accumulator in V16: the supplied initial value, or lane 0 when there is none.
            let start = match init {
                Some(acc) => {
                    self.materialize_fp(acc, V16);
                    0
                }
                None => {
                    self.emit_mem_fp(true, fp, V16, SP, off);
                    1
                }
            };
            for i in start..count {
                self.emit_mem_fp(true, fp, V17, SP, off + i * es);
                self.emit(Inst::FpDataProc2 { op: fpop, size: fp, rd: V16, rn: V16, rm: V17 });
            }
            return self.spill_fp(V16, result_ty);
        }

        // Integer reduction folded in X10. Signed min/max need their lanes sign-extended before
        // comparison (loads zero-extend); add/mul/bitwise only care about the low `elem` bits.
        let msize = mem_size(self.cx, elem);
        let need_sext = signed && matches!(op, SimdReduce::Max | SimdReduce::Min);
        let start = match init {
            Some(acc) => {
                self.materialize(acc, X10);
                0
            }
            None => {
                self.emit_mem_gpr(true, false, msize, X10, SP, off);
                if need_sext {
                    self.sign_extend_reg(X10, es);
                }
                1
            }
        };
        for i in start..count {
            self.emit_mem_gpr(true, false, msize, X11_HACK, SP, off + i * es);
            if need_sext {
                self.sign_extend_reg(X11_HACK, es);
            }
            match op {
                SimdReduce::Add => self.emit(Inst::AddSubReg {
                    op: AddSub::Add,
                    size: OperandSize::S64,
                    set_flags: false,
                    rd: X10,
                    rn: X10,
                    rm: X11_HACK,
                    amount: 0,
                }),
                SimdReduce::Mul => self.emit(Inst::Madd {
                    size: OperandSize::S64,
                    rd: X10,
                    rn: X10,
                    rm: X11_HACK,
                    ra: ZR,
                }),
                SimdReduce::And | SimdReduce::Or | SimdReduce::Xor => {
                    let logic = match op {
                        SimdReduce::And => LogicOp::And,
                        SimdReduce::Or => LogicOp::Orr,
                        _ => LogicOp::Eor,
                    };
                    self.emit(Inst::Logical {
                        op: logic,
                        size: OperandSize::S64,
                        rd: X10,
                        rn: X10,
                        rm: X11_HACK,
                        amount: 0,
                    });
                }
                SimdReduce::Max | SimdReduce::Min => {
                    // Keep the accumulator (X10) when it already wins, else take the lane (X11).
                    let cond = match (op, signed) {
                        (SimdReduce::Max, true) => Cond::Ge,
                        (SimdReduce::Max, false) => Cond::Hs,
                        (SimdReduce::Min, true) => Cond::Le,
                        (SimdReduce::Min, false) => Cond::Ls,
                        _ => unreachable!(),
                    };
                    self.emit(Inst::AddSubReg {
                        op: AddSub::Sub,
                        size: OperandSize::S64,
                        set_flags: true,
                        rd: ZR,
                        rn: X10,
                        rm: X11_HACK,
                        amount: 0,
                    });
                    self.emit(Inst::CondSel {
                        op: CondSel::Csel,
                        size: OperandSize::S64,
                        rd: X10,
                        rn: X10,
                        rm: X11_HACK,
                        cond,
                    });
                }
            }
        }
        self.spill(X10, result_ty)
    }

    /// `simd_saturating_add`/`simd_saturating_sub`: lane-wise saturating add/subtract, reusing the
    /// scalar saturating lowering (which clamps to the lane type's bounds) per lane. Integer lanes.
    fn emit_simd_saturating(&mut self, a: Value, b: Value, is_add: bool, signed: bool) -> Value {
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
            let av = Value::Slot { off: aoff + i * es, ty: elem };
            let bv = Value::Slot { off: boff + i * es, ty: elem };
            let r = self.emit_saturating(av, bv, is_add, signed, elem);
            self.materialize(r, X10);
            self.emit_mem_gpr(false, false, msize, X10, SP, roff + i * es);
        }
        Value::Slot { off: roff, ty: vec_ty }
    }

    /// `simd_ctpop`/`simd_ctlz`/`simd_cttz`/`simd_bswap`/`simd_bitreverse`: lane-wise integer bit
    /// operations, each reusing the scalar lowering per lane. The population/zero counts come back
    /// 64-bit and are narrowed to the lane width.
    fn emit_simd_bit_unary(&mut self, op: SimdBitOp, v: Value) -> Value {
        let vec_ty = v.ty();
        let (elem, count, es) = self.vector_info(vec_ty);
        let Some(off) = self.vector_to_slot(v) else {
            return Value::Undef { ty: vec_ty };
        };
        let (size, align) = self.cx.type_size_align(vec_ty);
        let roff = self.alloc_slot(size, align);
        let msize = mem_size(self.cx, elem);
        for i in 0..count {
            let lane = Value::Slot { off: off + i * es, ty: elem };
            let r = match op {
                SimdBitOp::Ctpop => {
                    let c = self.emit_ctpop(lane);
                    self.intcast(c, elem, false)
                }
                SimdBitOp::Ctlz => {
                    let c = self.emit_ctlz(lane);
                    self.intcast(c, elem, false)
                }
                SimdBitOp::Cttz => {
                    let c = self.emit_cttz(lane);
                    self.intcast(c, elem, false)
                }
                SimdBitOp::Bswap => self.emit_reverse(lane, DataProc1::Rev),
                SimdBitOp::Bitreverse => self.emit_reverse(lane, DataProc1::Rbit),
            };
            self.materialize(r, X10);
            self.emit_mem_gpr(false, false, msize, X10, SP, roff + i * es);
        }
        Value::Slot { off: roff, ty: vec_ty }
    }

    /// `simd_shuffle`: build a result vector by gathering lanes from the concatenation `x ++ y` at
    /// the (runtime-read) indices in `idx`. Implemented for byte lanes: the inputs are laid out
    /// contiguously in a scratch buffer and each result lane is a register-indexed byte load.
    fn emit_simd_shuffle(&mut self, x: Value, y: Value, idx: Value, result_ty: Type) -> Value {
        let (_elem, n, es) = self.vector_info(x.ty());
        let (_, out_n, _) = self.vector_info(result_ty);
        let (idx_elem, _, idx_es) = self.vector_info(idx.ty());
        let xoff = self.vector_to_slot(x);
        let yoff = self.vector_to_slot(y);
        let ioff = self.vector_to_slot(idx);
        let (Some(xoff), Some(yoff), Some(ioff)) = (xoff, yoff, ioff) else {
            return Value::Undef { ty: result_ty };
        };
        let lane = mem_size_from_bytes(es);
        let shift = es.trailing_zeros() as u8; // log2(element size); lanes are power-of-two sized
        // Concatenate the inputs into a `2*n`-lane buffer so an index in `0..2*n` selects a lane.
        let buf_off = self.alloc_slot(2 * n * es, es);
        for j in 0..(n * es) {
            self.emit_mem_gpr(true, false, MemSize::B, X10, SP, xoff + j);
            self.emit_mem_gpr(false, false, MemSize::B, X10, SP, buf_off + j);
            self.emit_mem_gpr(true, false, MemSize::B, X10, SP, yoff + j);
            self.emit_mem_gpr(false, false, MemSize::B, X10, SP, buf_off + n * es + j);
        }
        let (out_size, out_align) = self.cx.type_size_align(result_ty);
        let res_off = self.alloc_slot(out_size, out_align);
        let idxmsize = mem_size(self.cx, idx_elem);
        for i in 0..out_n {
            self.emit_mem_gpr(true, false, idxmsize, X12, SP, ioff + i * idx_es);
            self.emit_frame_addr(X13, buf_off);
            // addr = buf + (index << log2(es))
            self.emit(Inst::AddSubReg {
                op: AddSub::Add,
                size: OperandSize::S64,
                set_flags: false,
                rd: X13,
                rn: X13,
                rm: X12,
                amount: shift,
            });
            self.emit(Inst::LoadStoreUImm {
                load: true,
                signed: false,
                size: lane,
                rt: X10,
                rn: X13,
                offset: 0,
            });
            self.emit_mem_gpr(false, false, lane, X10, SP, res_off + i * es);
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

    /// `simd_insert`: return a copy of vector `x` with lane `idx` overwritten by scalar `val`.
    /// (`result_ty` is the same vector type as `x`.)
    fn emit_simd_insert(&mut self, x: Value, idx: u64, val: Value, result_ty: Type) -> Value {
        let (elem, count, es) = self.vector_info(x.ty());
        let Some(xoff) = self.vector_to_slot(x) else {
            return Value::Undef { ty: result_ty };
        };
        let (size, align) = self.cx.type_size_align(result_ty);
        let roff = self.alloc_slot(size, align);
        // Copy every lane across first (bit-preserving integer moves work for float lanes too).
        let msize = mem_size(self.cx, elem);
        for i in 0..count {
            self.emit_mem_gpr(true, false, msize, X10, SP, xoff + i * es);
            self.emit_mem_gpr(false, false, msize, X10, SP, roff + i * es);
        }
        // Overwrite the selected lane with the inserted scalar.
        let lane_off = roff + idx * es;
        if type_is_float(self.cx, elem) {
            self.materialize_fp(val, V16);
            self.emit_mem_fp(false, fp_size(self.cx, elem), V16, SP, lane_off);
        } else {
            self.materialize(val, X10);
            self.emit_mem_gpr(false, false, msize, X10, SP, lane_off);
        }
        Value::Slot { off: roff, ty: result_ty }
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

    type DIScope = gimli::write::UnitEntryId;
    type DILocation = crate::mach::inst::DebugLoc;
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
        // 128-bit integers are handled as a low/high word pair by dedicated paths in the builder,
        // which never call this. Reaching here with one means an i128 operation was not routed to
        // its 128-bit path (e.g. a `match` on an i128 value); fail loudly rather than silently
        // truncating to a single 64-bit register.
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

/// Map a lane-comparison condition to a NEON compare: the instruction, whether to swap the operands
/// (the `<`/`<=` forms are expressed as a reversed `>`/`>=`), and whether to invert the result (`ne`
/// is `cmeq` + `not`). Float lanes use the unsigned condition forms (floats are not "signed" here).
/// Returns `None` for conditions without a direct NEON form, leaving the scalar fallback to handle
/// them.
fn simd_cmp_neon(cond: Cond, is_float: bool) -> Option<(SimdOp, bool, bool)> {
    Some(if is_float {
        match cond {
            Cond::Eq => (SimdOp::Fcmeq, false, false),
            Cond::Ne => (SimdOp::Fcmeq, false, true),
            Cond::Hi => (SimdOp::Fcmgt, false, false), // a > b
            Cond::Hs => (SimdOp::Fcmge, false, false), // a >= b
            Cond::Lo => (SimdOp::Fcmgt, true, false),  // a < b  == b > a
            Cond::Ls => (SimdOp::Fcmge, true, false),  // a <= b == b >= a
            _ => return None,
        }
    } else {
        match cond {
            Cond::Eq => (SimdOp::Cmeq, false, false),
            Cond::Ne => (SimdOp::Cmeq, false, true),
            Cond::Gt => (SimdOp::Cmgt, false, false),
            Cond::Ge => (SimdOp::Cmge, false, false),
            Cond::Lt => (SimdOp::Cmgt, true, false), // a < b  == b > a
            Cond::Le => (SimdOp::Cmge, true, false), // a <= b == b >= a
            Cond::Hi => (SimdOp::Cmhi, false, false),
            Cond::Hs => (SimdOp::Cmhs, false, false),
            Cond::Lo => (SimdOp::Cmhi, true, false), // a <u b == b >u a
            Cond::Ls => (SimdOp::Cmhs, true, false), // a <=u b == b >=u a
            _ => return None,
        }
    })
}

/// Map a scalar floating-point unary op to its NEON two-register-misc equivalent, for the lane-wise
/// SIMD intrinsics. Every rounding/abs/neg/sqrt op has a vector form.
fn fp_op1_to_simd(op: FpOp1) -> Option<SimdUnOp> {
    Some(match op {
        FpOp1::Fabs => SimdUnOp::Fabs,
        FpOp1::Fneg => SimdUnOp::Fneg,
        FpOp1::Fsqrt => SimdUnOp::Fsqrt,
        FpOp1::Frintn => SimdUnOp::Frintn,
        FpOp1::Frintp => SimdUnOp::Frintp,
        FpOp1::Frintm => SimdUnOp::Frintm,
        FpOp1::Frintz => SimdUnOp::Frintz,
        FpOp1::Frinta => SimdUnOp::Frinta,
    })
}

/// Floating-point operand size for a float backend type (`f16` -> half, `f32` -> single, otherwise
/// double). `f128` has no hardware FP register form and is handled by libcalls before this is hit.
fn fp_size(cx: &CodegenCx<'_>, ty: Type) -> FpSize {
    match cx.type_data(ty) {
        TypeData::Float(16) => FpSize::S16,
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

impl ArgAssign {
    /// Reserve space for a stack argument of natural `size`/`align` bytes, returning the byte offset
    /// (within the outgoing-argument area) to place it at. Apple AArch64 packs stack arguments at
    /// their natural alignment and size; it does *not* round each up to an 8-byte slot the way the
    /// standard AAPCS64 does (so e.g. two `u32` stack args sit at offsets 0 and 4, not 0 and 8).
    fn stack_slot(&mut self, size: u32, align: u32) -> u32 {
        let off = (self.nsaa + align - 1) & !(align - 1);
        self.nsaa = off + size;
        off
    }
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
            let (size, align) = cx.type_size_align(ty);
            ParamLoc::Stack(a.stack_slot(size as u32, align as u32))
        }
    } else if a.ngrn < 8 {
        let loc = ParamLoc::Gpr(Gpr::from_encoding(a.ngrn));
        a.ngrn += 1;
        loc
    } else {
        let (size, align) = cx.type_size_align(ty);
        ParamLoc::Stack(a.stack_slot(size as u32, align as u32))
    };
    params.push((loc, ty));
}

/// One physical register's worth of a `PassMode::Cast` value: whether it is a SIMD&FP register, its
/// width in bytes (the register's natural width — 4 or 8), the number of meaningful bytes it carries
/// (`<= width`; smaller only for the final chunk of a non-8-multiple integer cast), and the byte
/// offset of those bytes within the aggregate.
#[derive(Clone, Copy)]
struct CastReg {
    fp: bool,
    width: u32,
    data: u32,
    offset: u32,
}

/// Enumerate the registers a `CastTarget` occupies, following AAPCS64: the prefix registers (at
/// consecutive offsets, or `prefix[0]@0` plus `rest@rest_offset`), then `ceil(rest.total / unit)`
/// copies of the `rest` unit. Every register is kept at its full width so the all-or-nothing
/// register/stack placement can be computed; `data` records how many bytes are actually meaningful.
fn cast_regs(cast: &CastTarget) -> Vec<CastReg> {
    fn is_fp(kind: RegKind) -> bool {
        matches!(kind, RegKind::Float | RegKind::Vector { .. })
    }
    let mut regs = Vec::new();
    let mut offset: u32 = 0;
    if let Some(ro) = cast.rest_offset {
        let p = cast.prefix[0];
        let w = p.size.bytes() as u32;
        regs.push(CastReg { fp: is_fp(p.kind), width: w, data: w, offset: 0 });
        offset = ro.bytes() as u32;
    } else {
        for p in &cast.prefix {
            let w = p.size.bytes() as u32;
            regs.push(CastReg { fp: is_fp(p.kind), width: w, data: w, offset });
            offset += w;
        }
    }
    let unit = cast.rest.unit;
    let unit_fp = is_fp(unit.kind);
    // An integer cast piece occupies a single 8-byte GPR, so a wider integer unit — e.g. the `i128`
    // rustc uses for a 16-byte aggregate, or for a `repr(align(16))` struct whose data is smaller —
    // must be split across multiple GPRs. (FP units are already <= 8 bytes: `s`/`d` registers.)
    let usz = if unit_fp {
        (unit.size.bytes() as u32).max(1)
    } else {
        (unit.size.bytes() as u32).clamp(1, 8)
    };
    let total = cast.rest.total.bytes() as u32;
    if total > 0 {
        let n = total.div_ceil(usz);
        for i in 0..n {
            let rel = i * usz;
            let data = (total - rel).min(usz);
            regs.push(CastReg { fp: unit_fp, width: usz, data, offset: offset + rel });
        }
    }
    // FP cast pieces map to a single SIMD&FP register (s/d); 128-bit vector HFA lanes are not
    // produced by Rust's C ABI and would mis-size below, so make that loud.
    debug_assert!(regs.iter().all(|r| !r.fp || r.width <= 8), "FP cast piece wider than 8 bytes");
    regs
}

/// Assign a `PassMode::Cast` argument, pushing one physical parameter per register it occupies.
/// AAPCS64 placement is all-or-nothing per register bank: if the cast's registers do not all fit in
/// the remaining registers of their bank, the whole aggregate is passed contiguously on the stack.
fn push_cast_param<'tcx>(
    cx: &CodegenCx<'tcx>,
    params: &mut Vec<(ParamLoc, Type)>,
    a: &mut ArgAssign,
    cast: &CastTarget,
) {
    let regs = cast_regs(cast);
    let piece_ty = |r: &CastReg| {
        if r.fp {
            cx.intern_type(TypeData::Float(r.width * 8))
        } else {
            // Spill an integer piece at its meaningful width (rounded to a natural load/store size)
            // so a 4-byte chunk uses `str w`, not `str x`; only `data` bytes are ever consumed.
            cx.intern_type(TypeData::Int(r.data.next_power_of_two().max(1) * 8))
        }
    };
    let n_int = regs.iter().filter(|r| !r.fp).count() as u8;
    let n_fp = regs.iter().filter(|r| r.fp).count() as u8;
    if a.ngrn + n_int <= 8 && a.nsrn + n_fp <= 8 {
        for r in &regs {
            if r.fp {
                params.push((ParamLoc::Fp(Vreg::from_encoding(a.nsrn)), piece_ty(r)));
                a.nsrn += 1;
            } else {
                params.push((ParamLoc::Gpr(Gpr::from_encoding(a.ngrn)), piece_ty(r)));
                a.ngrn += 1;
            }
        }
    } else {
        if n_int > 0 {
            a.ngrn = 8;
        }
        if n_fp > 0 {
            a.nsrn = 8;
        }
        // A stack-passed cast occupies its register pieces back-to-back (each piece is a
        // register-sized chunk), aligned to the aggregate's natural alignment. The footprint is the
        // pieces' total width, *not* the data size rounded to 8 — a sub-register cast (e.g. an
        // `i32`-unit 4-byte union) takes a 4-byte slot, while LLVM agrees.
        let footprint = regs.iter().map(|r| r.offset + r.width).max().unwrap_or(0);
        let align = cast.align(cx).bytes() as u32;
        let base = a.stack_slot(footprint, align);
        for r in &regs {
            params.push((ParamLoc::Stack(base + r.offset), piece_ty(r)));
        }
    }
}
fn build_param_list<'tcx>(cx: &CodegenCx<'tcx>, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> ParamList {
    let mut params = Vec::new();
    let ptr = cx.intern_type(TypeData::Ptr);
    if fn_abi.ret.is_indirect() {
        // The indirect-return (sret) pointer arrives in x8 (never on the stack).
        params.push((ParamLoc::Gpr(Gpr::from_encoding(8)), ptr));
    }
    let mut a = ArgAssign::default();
    for (argn, arg) in fn_abi.args.iter().enumerate() {
        // Apple AArch64 passes every variadic argument on the stack; exhaust the register banks
        // once the variadic portion begins so the outgoing-argument-area sizing accounts for them.
        if fn_abi.c_variadic && argn == fn_abi.fixed_count as usize {
            a.ngrn = 8;
            a.nsrn = 8;
        }
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
            PassMode::Indirect { meta_attrs, .. } => {
                // A sized indirect argument is a single (data) pointer. An *unsized* indirect
                // argument is a wide pointer — the data pointer plus its metadata (slice length or
                // vtable pointer) — and occupies two consecutive parameters, matching how the SSA
                // driver reads it back (`get_param(i)`, `get_param(i + 1)`).
                push_scalar_param(cx, &mut params, &mut a, ptr);
                if meta_attrs.is_some() {
                    push_scalar_param(cx, &mut params, &mut a, ptr);
                }
            }
            PassMode::Cast { ref cast, .. } => push_cast_param(cx, &mut params, &mut a, cast),
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
                if matches!(cx.type_data(ty), TypeData::Float(128)) {
                    // `f128` arrives in a full 128-bit `q` register.
                    push_mem_q(out, false, reg, SP, off);
                } else {
                    push_mem_fp(out, false, fp_size(cx, ty), reg, SP, off);
                }
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
                    if matches!(cx.type_data(ty), TypeData::Float(128)) {
                        push_mem_q(out, true, V16, FP, incoming);
                        push_mem_q(out, false, V16, SP, off);
                    } else {
                        let size = fp_size(cx, ty);
                        push_mem_fp(out, true, size, V16, FP, incoming);
                        push_mem_fp(out, false, size, V16, SP, off);
                    }
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
                if self.cx.got_syms.borrow().contains(&sym) {
                    // Foreign static: load its address from the GOT, then add any field offset.
                    let base = SymRef { name: self.cx.sym_name(sym), addend: 0 };
                    self.emit(Inst::AdrpGot { rd: reg, sym: base.clone() });
                    self.emit(Inst::LdrGotLo { rt: reg, rn: reg, sym: base });
                    if offset != 0 {
                        self.load_imm(X16, offset as u128, OperandSize::S64);
                        self.emit(Inst::AddSubReg { op: AddSub::Add, size: OperandSize::S64, set_flags: false, rd: reg, rn: reg, rm: X16, amount: 0 });
                    }
                } else {
                    self.emit(Inst::Adrp { rd: reg, sym: symref.clone() });
                    self.emit(Inst::AddLo { rd: reg, rn: reg, sym: symref });
                }
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

    /// Whether a backend type is `f128`, the IEEE binary128 quad-precision float. It has no AArch64
    /// hardware register form; every operation is a `compiler_builtins` libcall over a 16-byte
    /// value held in a `q` register.
    fn is_f128(&self, ty: Type) -> bool {
        matches!(self.cx.type_data(ty), TypeData::Float(128))
    }

    /// `f128` absolute value: clear the sign bit (bit 127, the MSB of the high word); the low word
    /// is unchanged. No libcall needed.
    fn f128_fabs(&mut self, arg: Value) -> Value {
        let ty = arg.ty();
        self.materialize128(arg, X9, X10);
        self.load_imm(X11_HACK, 0x7FFF_FFFF_FFFF_FFFF, OperandSize::S64);
        self.emit(Inst::Logical {
            op: LogicOp::And,
            size: OperandSize::S64,
            rd: X10,
            rn: X10,
            rm: X11_HACK,
            amount: 0,
        });
        self.spill128(X9, X10, ty)
    }

    /// `f128` negation: flip the sign bit (bit 127), which lives in the high 64-bit word. A native
    /// `fneg` would treat the value as `f64` and corrupt it, so operate on the two GPR words.
    fn f128_fneg(&mut self, arg: Value) -> Value {
        let ty = arg.ty();
        self.materialize128(arg, X9, X10);
        self.load_imm(X11_HACK, 0x8000_0000_0000_0000, OperandSize::S64);
        self.emit(Inst::Logical {
            op: LogicOp::Eor,
            size: OperandSize::S64,
            rd: X10,
            rn: X10,
            rm: X11_HACK,
            amount: 0,
        });
        self.spill128(X9, X10, ty)
    }

    /// `f128` `copysign(x, y)`: x's magnitude with y's sign. Both the sign bit and the magnitude
    /// live in/around the high word, so combine `x_hi & ~signbit` with `y_hi & signbit`; there is no
    /// `f128` libm `copysign` on macOS, so this is done inline on the GPR words.
    fn f128_copysign(&mut self, x: Value, y: Value) -> Value {
        let ty = x.ty();
        self.materialize128(x, X9, X10); // X9 = x lo, X10 = x hi
        self.materialize128(y, X12, X13); // X13 = y hi (X12 lo unused)
        // Clear x's sign bit (high word & 0x7FFF...).
        self.load_imm(X11_HACK, 0x7FFF_FFFF_FFFF_FFFF, OperandSize::S64);
        self.emit(Inst::Logical {
            op: LogicOp::And,
            size: OperandSize::S64,
            rd: X10,
            rn: X10,
            rm: X11_HACK,
            amount: 0,
        });
        // Isolate y's sign bit (high word & 0x8000...).
        self.load_imm(X11_HACK, 0x8000_0000_0000_0000, OperandSize::S64);
        self.emit(Inst::Logical {
            op: LogicOp::And,
            size: OperandSize::S64,
            rd: X13,
            rn: X13,
            rm: X11_HACK,
            amount: 0,
        });
        // Combine: x_hi |= y_sign.
        self.emit(Inst::Logical {
            op: LogicOp::Orr,
            size: OperandSize::S64,
            rd: X10,
            rn: X10,
            rm: X13,
            amount: 0,
        });
        self.spill128(X9, X10, ty)
    }

    /// Load a 16-byte `f128` value into the `q` register `qreg`.
    fn materialize_q(&mut self, val: Value, qreg: Vreg) {
        match val {
            Value::Slot { off, .. } => self.emit_q(true, qreg, off),
            Value::Const { bits, .. } => {
                // Stage the 16-byte bit pattern in a temp slot, then load it into the q register.
                let off = self.alloc_slot(16, 16);
                self.load_imm(X9, bits, OperandSize::S64);
                self.emit_mem_gpr(false, false, MemSize::X, X9, SP, off);
                self.load_imm(X9, bits >> 64, OperandSize::S64);
                self.emit_mem_gpr(false, false, MemSize::X, X9, SP, off + 8);
                self.emit_q(true, qreg, off);
            }
            Value::Sym { sym, offset, .. } => {
                let symref = SymRef { name: self.cx.sym_name(sym), addend: offset };
                self.emit(Inst::Adrp { rd: X9, sym: symref.clone() });
                self.emit(Inst::AddLo { rd: X9, rn: X9, sym: symref });
                self.emit(Inst::LoadStoreQ { load: true, rt: qreg, rn: X9, offset: 0 });
            }
            Value::Undef { .. } => {
                let off = self.alloc_slot(16, 16);
                self.emit_mem_gpr(false, false, MemSize::X, ZR, SP, off);
                self.emit_mem_gpr(false, false, MemSize::X, ZR, SP, off + 8);
                self.emit_q(true, qreg, off);
            }
        }
    }

    /// Store the `q` register `qreg` (a 16-byte `f128`) into a fresh frame slot.
    fn spill_q_val(&mut self, qreg: Vreg, ty: Type) -> Value {
        let off = self.alloc_slot(16, 16);
        self.emit_q(false, qreg, off);
        Value::Slot { off, ty }
    }

    /// An `f128` binary arithmetic op via a `compiler_builtins` libcall (`__addtf3`/`__subtf3`/
    /// `__multf3`/`__divtf3`): operands in `q0`/`q1`, result in `q0`.
    fn f128_binop(&mut self, sym: &str, lhs: Value, rhs: Value) -> Value {
        let ty = lhs.ty();
        self.materialize_q(lhs, V0);
        self.materialize_q(rhs, V1);
        self.emit(Inst::Bl { sym: SymRef::new(sym) });
        self.spill_q_val(V0, ty)
    }

    /// An `f128` comparison via a `compiler_builtins` libcall (`__eqtf2`/`__lttf2`/...): the call
    /// returns an `i32` in `w0` whose sign/zero relationship to `0` (using `cond`) gives the result.
    fn f128_cmp(&mut self, sym: &str, cond: Cond, lhs: Value, rhs: Value) -> Value {
        self.materialize_q(lhs, V0);
        self.materialize_q(rhs, V1);
        self.emit(Inst::Bl { sym: SymRef::new(sym) });
        self.emit(Inst::AddSubImm {
            op: AddSub::Sub,
            size: OperandSize::S32,
            set_flags: true,
            rd: ZR,
            rn: X0,
            imm12: 0,
            shift12: false,
        });
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

    /// Convert an integer to `f128` via a libcall (`__float[un]ditf` for <=64-bit sources, promoted
    /// to 64 bits first, or `__float[un]titf` for `i128`). Result in `q0`.
    fn int_to_f128(&mut self, signed: bool, val: Value, dest_ty: Type) -> Value {
        if self.is_int128(val.ty()) {
            let sym = if signed { "___floattitf" } else { "___floatuntitf" };
            self.materialize128(val, X0, X1);
            self.emit(Inst::Bl { sym: SymRef::new(sym) });
            return self.spill_q_val(V0, dest_ty);
        }
        let i64ty = self.cx.intern_type(TypeData::Int(64));
        let wide = if signed { self.sext(val, i64ty) } else { self.zext(val, i64ty) };
        let sym = if signed { "___floatditf" } else { "___floatunditf" };
        self.materialize(wide, X0);
        self.emit(Inst::Bl { sym: SymRef::new(sym) });
        self.spill_q_val(V0, dest_ty)
    }

    /// Convert `f128` to an integer via a libcall (`__fix[uns]tfti` for `i128`, else `__fix[uns]tfdi`
    /// to `i64` then clamped to the destination width). These saturate and map NaN to zero.
    fn f128_to_int(&mut self, signed: bool, val: Value, dest_ty: Type) -> Value {
        if self.is_int128(dest_ty) {
            let sym = if signed { "___fixtfti" } else { "___fixunstfti" };
            self.materialize_q(val, V0);
            self.emit(Inst::Bl { sym: SymRef::new(sym) });
            return self.spill128(X0, X1, dest_ty);
        }
        let sym = if signed { "___fixtfdi" } else { "___fixunstfdi" };
        self.materialize_q(val, V0);
        self.emit(Inst::Bl { sym: SymRef::new(sym) });
        self.clamp_fp_to_int(X0, dest_ty, signed)
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

    /// 128-bit overflow-checked add/subtract: returns `(result, overflowed)`. The carry chain sets
    /// the flags on the high word; unsigned overflow is the carry-out (`hs`) for add / borrow
    /// (`lo`) for subtract, and signed overflow is the `V` flag (`vs`).
    fn checked128_addsub(&mut self, oop: OverflowOp, signed: bool, lhs: Value, rhs: Value) -> (Value, Value) {
        let ty = lhs.ty();
        let bool_ty = self.cx.intern_type(TypeData::Int(1));
        let op = match oop {
            OverflowOp::Add => AddSub::Add,
            OverflowOp::Sub => AddSub::Sub,
            OverflowOp::Mul => unreachable!("128-bit checked mul is handled separately"),
        };
        self.materialize128(lhs, X9, X10);
        self.materialize128(rhs, X11, X12);
        self.emit(Inst::AddSubReg { op, size: OperandSize::S64, set_flags: true, rd: X9, rn: X9, rm: X11, amount: 0 });
        self.emit(Inst::AddSubCarry { op, size: OperandSize::S64, set_flags: true, rd: X10, rn: X10, rm: X12 });
        let cond = match (op, signed) {
            (AddSub::Add, false) => Cond::Hs,
            (AddSub::Sub, false) => Cond::Lo,
            (_, true) => Cond::Vs,
        };
        // Read the overflow flag before spilling the result words (stores do not affect NZCV, but
        // reading it first keeps the dependency obvious).
        self.emit(Inst::CondSel { op: CondSel::Csinc, size: OperandSize::S32, rd: X13, rn: ZR, rm: ZR, cond: cond.invert() });
        let result = self.spill128(X9, X10, ty);
        let overflow = self.spill(X13, bool_ty);
        (result, overflow)
    }

    /// 128-bit overflow-checked multiply, returning `(low_product, overflowed)`.
    ///
    /// The signed case calls the purpose-built `__muloti4(a, b, &overflow)` libcall. The unsigned
    /// case computes the wrapping product and uses the standard idiom `a != 0 && (a*b)/a != b`,
    /// where the division (a libcall) is guarded against `a == 0` by substituting `1` (whose result
    /// is masked off by the `a != 0` term).
    fn checked128_mul(&mut self, signed: bool, lhs: Value, rhs: Value) -> (Value, Value) {
        let ty = lhs.ty();
        if signed {
            let i32t = self.cx.intern_type(TypeData::Int(32));
            let ovf_off = self.alloc_slot(4, 4);
            self.materialize128(lhs, X0, X1);
            self.materialize128(rhs, X2, X3);
            self.emit_frame_addr(X4, ovf_off);
            self.emit(Inst::Bl { sym: SymRef::new("___muloti4") });
            let result = self.spill128(X0, X1, ty);
            let ovf = Value::Slot { off: ovf_off, ty: i32t };
            let zero = self.cx.const_uint(i32t, 0);
            let overflow = self.icmp(IntPredicate::IntNE, ovf, zero);
            (result, overflow)
        } else {
            let zero = Value::Const { bits: 0, ty };
            let one = Value::Const { bits: 1, ty };
            let prod = self.int128_mul(lhs, rhs);
            let a_is_zero = self.icmp(IntPredicate::IntEQ, lhs, zero);
            let a_safe = self.select(a_is_zero, one, lhs);
            let q = self.int128_bin_libcall("___udivti3", prod, a_safe);
            let q_ne_b = self.icmp(IntPredicate::IntNE, q, rhs);
            let a_nonzero = self.icmp(IntPredicate::IntNE, lhs, zero);
            let overflow = self.and(q_ne_b, a_nonzero);
            (prod, overflow)
        }
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

    /// Lower the `rotate_left`/`rotate_right` intrinsics as `(x << k) | (x >>u (W - k))`. The shift
    /// (a `u32`) is widened/narrowed to the value's type and masked to `W - 1`; the complement
    /// `(W - k) & (W - 1)` is masked too so a zero rotate degenerates to `x | x` instead of an
    /// out-of-range shift. Works for every integer width — sub-word results are masked by the narrow
    /// spill, and 128-bit values go through the shift libcalls. The right shift is always logical.
    fn emit_rotate(&mut self, x: Value, raw_shift: Value, left: bool) -> Value {
        let ty = x.ty();
        let bits = match self.cx.type_data(ty) {
            TypeData::Int(b) => b as u128,
            _ => 64,
        };
        let shift = self.intcast(raw_shift, ty, false);
        let mask = self.cx.const_uint(ty, (bits - 1) as u64);
        let k = self.and(shift, mask);
        let width = self.cx.const_uint(ty, bits as u64);
        let comp = self.sub(width, k);
        let comp = self.and(comp, mask);
        let (shl_amt, shr_amt) = if left { (k, comp) } else { (comp, k) };
        let lo = self.shl(x, shl_amt);
        let hi = self.lshr(x, shr_amt);
        self.or(lo, hi)
    }

    /// Convert a 128-bit integer to a float via a compiler-builtins libcall (`__float[un]tidf` /
    /// `__float[un]tisf`): the integer is in `x0:x1`, the result returns in `d0`/`s0`.
    fn int128_to_fp(&mut self, signed: bool, val: Value, dest_ty: Type) -> Value {
        let sym = match (fp_size(self.cx, dest_ty), signed) {
            (FpSize::S64, true) => "___floattidf",
            (FpSize::S64, false) => "___floatuntidf",
            (FpSize::S32, true) => "___floattisf",
            (FpSize::S32, false) => "___floatuntisf",
            (FpSize::S16, true) => "___floattihf",
            (FpSize::S16, false) => "___floatuntihf",
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
            (FpSize::S16, true) => "___fixhfti",
            (FpSize::S16, false) => "___fixunshfti",
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

    /// Split a 128-bit value into its low and high 64-bit words as separate `i64` values.
    fn int128_split(&mut self, v: Value) -> (Value, Value) {
        let i64t = self.cx.intern_type(TypeData::Int(64));
        self.materialize128(v, X9, X10);
        let lo = self.spill(X9, i64t);
        let hi = self.spill(X10, i64t);
        (lo, hi)
    }

    /// `ctpop` of a 128-bit value: the population counts of the two words summed (returned as a
    /// 64-bit count for the caller to narrow).
    fn emit_ctpop128(&mut self, v: Value) -> Value {
        let (lo, hi) = self.int128_split(v);
        let clo = self.emit_ctpop(lo);
        let chi = self.emit_ctpop(hi);
        self.add(clo, chi)
    }

    /// `ctlz` of a 128-bit value: `clz(hi)` unless the high word is zero, in which case
    /// `64 + clz(lo)` (and `ctlz(0) == 128` falls out).
    fn emit_ctlz128(&mut self, v: Value) -> Value {
        let i64t = self.cx.intern_type(TypeData::Int(64));
        let (lo, hi) = self.int128_split(v);
        let clz_lo = self.emit_ctlz(lo);
        let clz_hi = self.emit_ctlz(hi);
        let sixtyfour = self.cx.const_uint(i64t, 64);
        let lo_plus = self.add(clz_lo, sixtyfour);
        let zero = self.cx.const_uint(i64t, 0);
        let hi_is_zero = self.icmp(IntPredicate::IntEQ, hi, zero);
        self.select(hi_is_zero, lo_plus, clz_hi)
    }

    /// `cttz` of a 128-bit value: `cttz(lo)` unless the low word is zero, in which case
    /// `64 + cttz(hi)` (and `cttz(0) == 128` falls out).
    fn emit_cttz128(&mut self, v: Value) -> Value {
        let i64t = self.cx.intern_type(TypeData::Int(64));
        let (lo, hi) = self.int128_split(v);
        let ctz_lo = self.emit_cttz(lo);
        let ctz_hi = self.emit_cttz(hi);
        let sixtyfour = self.cx.const_uint(i64t, 64);
        let hi_plus = self.add(ctz_hi, sixtyfour);
        let zero = self.cx.const_uint(i64t, 0);
        let lo_is_zero = self.icmp(IntPredicate::IntEQ, lo, zero);
        self.select(lo_is_zero, hi_plus, ctz_lo)
    }

    /// `bswap`/`bitreverse` of a 128-bit value: reverse each 64-bit word and swap the two words
    /// (reversing the full 16 bytes / 128 bits).
    fn emit_reverse128(&mut self, v: Value, op: DataProc1) -> Value {
        let ty = v.ty();
        self.materialize128(v, X9, X10); // lo = X9, hi = X10
        self.emit(Inst::DataProc1 { op, size: OperandSize::S64, rd: X9, rn: X9 });
        self.emit(Inst::DataProc1 { op, size: OperandSize::S64, rd: X10, rn: X10 });
        // Reversed low word becomes the high word and vice versa.
        self.spill128(X10, X9, ty)
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
        // 128-bit operands can't go through the i64 path below (it would truncate them). Add/sub at
        // full width with hardware carry/overflow detection, then clamp to the type's bounds: signed
        // overflow saturates to MIN when `a < 0` (same-sign add or `a - b` going negative) else MAX;
        // unsigned add saturates to MAX on carry, unsigned sub to 0 on borrow.
        if n == 128 {
            let oop = if is_add { OverflowOp::Add } else { OverflowOp::Sub };
            let (s, overflow) = self.checked128_addsub(oop, signed, a, b);
            let bound = if signed {
                let zero = Value::Const { bits: 0, ty: result_ty };
                let min = Value::Const { bits: 1u128 << 127, ty: result_ty };
                let max = Value::Const { bits: (1u128 << 127) - 1, ty: result_ty };
                let a_neg = self.icmp(IntPredicate::IntSLT, a, zero);
                self.select(a_neg, min, max)
            } else if is_add {
                Value::Const { bits: u128::MAX, ty: result_ty }
            } else {
                Value::Const { bits: 0, ty: result_ty }
            };
            return self.select(overflow, bound, s);
        }
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
                    FpSize::S64 => OperandSize::S64,
                    FpSize::S16 | FpSize::S32 => OperandSize::S32,
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
        if self.is_f128(ty) {
            let sym = match op {
                FpOp2::Fadd => "___addtf3",
                FpOp2::Fsub => "___subtf3",
                FpOp2::Fmul => "___multf3",
                FpOp2::Fdiv => "___divtf3",
            };
            return self.f128_binop(sym, lhs, rhs);
        }
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
            // `f16` libm math is promoted to `f32` by the callers before reaching here.
            FpSize::S16 => unreachable!("f16 libm math is promoted to f32"),
        }
    }

    /// Call a unary libm routine, passing the argument in `d0`/`s0` and collecting the result.
    fn fp_libm_unary(&mut self, base: &str, arg: Value) -> Value {
        let ty = arg.ty();
        if matches!(fp_size(self.cx, ty), FpSize::S16) {
            // `f16` has no libm form; promote to `f32`, call the `f32` routine, and round the result
            // back to `f16` (matching LLVM, which legalizes `f16` libm calls by promoting to `f32`).
            self.materialize_fp(arg, V0);
            self.emit(Inst::FpCvt { from: FpSize::S16, to: FpSize::S32, rd: V0, rn: V0 });
            self.emit(Inst::Bl { sym: SymRef::new(format!("_{base}f")) });
            self.emit(Inst::FpCvt { from: FpSize::S32, to: FpSize::S16, rd: V0, rn: V0 });
            return self.spill_fp(V0, ty);
        }
        let sym = self.libm_symbol(base, ty);
        self.materialize_fp(arg, V0);
        self.emit(Inst::Bl { sym: SymRef::new(sym) });
        self.spill_fp(V0, ty)
    }

    /// Call a binary libm routine (`pow`, `copysign`), passing args in `d0`/`d1` (or `s0`/`s1`).
    fn fp_libm_binary(&mut self, base: &str, a: Value, b: Value) -> Value {
        let ty = a.ty();
        if matches!(fp_size(self.cx, ty), FpSize::S16) {
            // See `fp_libm_unary`: promote both `f16` args to `f32`, call the `f32` routine, demote.
            self.materialize_fp(a, V0);
            self.emit(Inst::FpCvt { from: FpSize::S16, to: FpSize::S32, rd: V0, rn: V0 });
            self.materialize_fp(b, V1);
            self.emit(Inst::FpCvt { from: FpSize::S16, to: FpSize::S32, rd: V1, rn: V1 });
            self.emit(Inst::Bl { sym: SymRef::new(format!("_{base}f")) });
            self.emit(Inst::FpCvt { from: FpSize::S32, to: FpSize::S16, rd: V0, rn: V0 });
            return self.spill_fp(V0, ty);
        }
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
        if matches!(fp_size(self.cx, ty), FpSize::S16) {
            // Promote the `f16` base to `f32`, call `__powisf2`, and round back to `f16`.
            self.materialize_fp(base_val, V0);
            self.emit(Inst::FpCvt { from: FpSize::S16, to: FpSize::S32, rd: V0, rn: V0 });
            self.materialize(exp, X0);
            self.emit(Inst::Bl { sym: SymRef::new("___powisf2") });
            self.emit(Inst::FpCvt { from: FpSize::S32, to: FpSize::S16, rd: V0, rn: V0 });
            return self.spill_fp(V0, ty);
        }
        let sym = match fp_size(self.cx, ty) {
            FpSize::S64 => "___powidf2",
            FpSize::S32 => "___powisf2",
            FpSize::S16 => unreachable!("f16 powi is promoted to f32"),
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
/// the two-condition (`ONE`/`UEQ`) and constant (`False`/`True`) predicates are handled directly in
/// `fcmp` and never reach here.
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
        RealPredicate::RealONE
        | RealPredicate::RealUEQ
        | RealPredicate::RealPredicateFalse
        | RealPredicate::RealPredicateTrue => {
            unreachable!("two-condition / constant float predicates are handled in `fcmp`")
        }
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
        _dbg_loc: crate::mach::inst::DebugLoc,
        _variable_alloca: Value,
        _direct_offset: Size,
        _indirect_offsets: &[Size],
        _fragment: &Option<std::ops::Range<Size>>,
    ) {
    }
    fn dbg_var_value(
        &mut self,
        _dbg_var: (),
        _dbg_loc: crate::mach::inst::DebugLoc,
        _value: Value,
        _direct_offset: Size,
        _indirect_offsets: &[Size],
        _fragment: &Option<std::ops::Range<Size>>,
    ) {
    }
    fn set_dbg_loc(&mut self, dbg_loc: crate::mach::inst::DebugLoc) {
        // Record a source-location marker in the instruction stream; layout turns it into a
        // `.debug_line` row. No-op when debug info is disabled (the marker is simply never emitted
        // because `set_dbg_loc` is only called by the SSA driver with debug info enabled).
        self.emit(crate::mach::inst::Inst::DebugLoc(dbg_loc));
    }
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
            PassMode::Cast { ref cast, .. } => {
                // A cast aggregate arrives in one or more registers (integer GPRs for a composite,
                // SIMD&FP registers for an HFA). Each was spilled to its own param slot; copy the
                // meaningful bytes of each back into the destination aggregate at its true offset.
                let regs = cast_regs(cast);
                self.materialize(dst.val.llval, X9);
                let agg = dst.layout.size.bytes();
                for r in &regs {
                    let slot = self.get_param(*idx);
                    *idx += 1;
                    let Value::Slot { off, .. } = slot else { continue };
                    let n = (r.data as u64).min(agg.saturating_sub(r.offset as u64));
                    if n > 0 {
                        self.copy_slot_to_base_off(off, X9, r.offset as u64, n);
                    }
                }
            }
        }
    }
    fn store_arg(
        &mut self,
        arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        val: Value,
        dst: PlaceRef<'tcx, Value>,
    ) {
        // A `PassMode::Cast` value (e.g. a call's small-aggregate/HFA return collected by
        // `collect_cast_ret`) is a slot already laid out as the aggregate; copy its bytes into the
        // destination place. Everything else is a scalar immediate.
        if let PassMode::Cast { .. } = arg_abi.mode {
            if let Value::Slot { off, .. } = val {
                self.materialize(dst.val.llval, X9);
                self.copy_slot_to_ptr(off, X9, dst.layout.size.bytes());
                return;
            }
        }
        OperandValue::Immediate(val).store(self, dst);
    }
}

impl<'a, 'tcx> IntrinsicCallBuilderMethods<'tcx> for Builder<'a, 'tcx> {
    fn codegen_intrinsic_call(
        &mut self,
        instance: Instance<'tcx>,
        args: &[OperandRef<'tcx, Value>],
        result_layout: TyAndLayout<'tcx>,
        result_place: Option<PlaceValue<Value>>,
        _span: Span,
    ) -> IntrinsicResult<'tcx, Value> {
        let name = self.cx.tcx.item_name(instance.def_id());
        match name {
            // `black_box` is an optimization barrier; with no optimizer it is the identity.
            sym::black_box => IntrinsicResult::Operand(args[0].val),
            // Population count: a SWAR sequence, narrowed to the `u32` result type.
            sym::ctpop => {
                let v = args[0].immediate();
                let count = if self.is_int128(v.ty()) { self.emit_ctpop128(v) } else { self.emit_ctpop(v) };
                let result_ty = self.cx.immediate_backend_type(result_layout);
                let count = self.intcast(count, result_ty, false);
                IntrinsicResult::Operand(OperandValue::Immediate(count))
            }
            // Bit rotation. AArch64 only rotates right (and only at 32/64-bit), so both directions
            // are lowered uniformly to a shift/or pair that handles every width including i128.
            sym::rotate_left | sym::rotate_right => {
                let left = name == sym::rotate_left;
                let r = self.emit_rotate(args[0].immediate(), args[1].immediate(), left);
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            // Count leading zeros (`ctlz_nonzero` shares the lowering; `clz` handles zero anyway).
            sym::ctlz | sym::ctlz_nonzero => {
                let v = args[0].immediate();
                let count = if self.is_int128(v.ty()) { self.emit_ctlz128(v) } else { self.emit_ctlz(v) };
                let result_ty = self.cx.immediate_backend_type(result_layout);
                let count = self.intcast(count, result_ty, false);
                IntrinsicResult::Operand(OperandValue::Immediate(count))
            }
            // Count trailing zeros, as `clz(rbit(x))` (the `_nonzero` form shares the lowering).
            sym::cttz | sym::cttz_nonzero => {
                let v = args[0].immediate();
                let count = if self.is_int128(v.ty()) { self.emit_cttz128(v) } else { self.emit_cttz(v) };
                let result_ty = self.cx.immediate_backend_type(result_layout);
                let count = self.intcast(count, result_ty, false);
                IntrinsicResult::Operand(OperandValue::Immediate(count))
            }
            // Byte reverse (`swap_bytes`) and bit reverse (`reverse_bits`).
            sym::bswap => {
                let v = args[0].immediate();
                let r = if self.is_int128(v.ty()) { self.emit_reverse128(v, DataProc1::Rev) } else { self.emit_reverse(v, DataProc1::Rev) };
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::bitreverse => {
                let v = args[0].immediate();
                let r = if self.is_int128(v.ty()) { self.emit_reverse128(v, DataProc1::Rbit) } else { self.emit_reverse(v, DataProc1::Rbit) };
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            // `ptr_mask(ptr, mask)` masks the address bits of a pointer: `(ptr.addr() & mask)` cast
            // back to a pointer. A plain 64-bit AND of the pointer and mask registers.
            sym::ptr_mask => {
                let ptr = args[0].immediate();
                let mask = args[1].immediate();
                let r = self.and(ptr, mask);
                IntrinsicResult::Operand(OperandValue::Immediate(r))
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
            // `raw_eq(a, b)`: bytewise equality of two `&T`, i.e. `memcmp(a, b, size_of::<T>()) == 0`.
            sym::raw_eq => {
                let t = instance.args.type_at(0);
                let size = self.cx.layout_of(t).size.bytes();
                self.materialize(args[0].immediate(), X0);
                self.materialize(args[1].immediate(), X1);
                self.load_imm(X2, size as u128, OperandSize::S64);
                self.emit(Inst::Bl { sym: SymRef::new("_memcmp") });
                let i32t = self.cx.intern_type(TypeData::Int(32));
                let cmp = self.spill(X0, i32t);
                let zero = self.cx.const_int(i32t, 0);
                let eq = self.icmp(IntPredicate::IntEQ, cmp, zero);
                IntrinsicResult::Operand(OperandValue::Immediate(eq))
            }
            // A volatile load through a pointer. A scalar (or scalar-pair) result is produced as
            // an operand directly; a non-scalar (aggregate) result must be copied into the caller's
            // result place — `OperandValue::Immediate` cannot represent an aggregate, and doing so
            // silently truncated the value to one register, corrupting the surrounding bytes (this
            // is how a `ptr::read_volatile::<MaybeUninit<T>>` of a large `T`, as used by
            // crossbeam-deque's work-stealing buffer, was miscompiled).
            sym::volatile_load | sym::unaligned_volatile_load => {
                let ptr = args[0].immediate();
                if self.cx.is_backend_immediate(result_layout) {
                    let ty = self.cx.immediate_backend_type(result_layout);
                    let load = self.volatile_load(ty, ptr);
                    IntrinsicResult::Operand(OperandValue::Immediate(load))
                } else if let BackendRepr::ScalarPair(a, b) = result_layout.backend_repr {
                    // A scalar pair (e.g. a wide pointer): load each half as its own scalar so the
                    // operand is a `Pair`, mirroring `load_operand`.
                    let b_offset = a.size(self.cx).align_to(b.align(self.cx).abi);
                    let ty_a = self.cx.scalar_pair_element_backend_type(result_layout, 0, true);
                    let ty_b = self.cx.scalar_pair_element_backend_type(result_layout, 1, true);
                    let off = self.const_usize(b_offset.bytes());
                    let ptr_b = self.inbounds_ptradd(ptr, off);
                    let a_val = self.volatile_load(ty_a, ptr);
                    let b_val = self.volatile_load(ty_b, ptr_b);
                    IntrinsicResult::Operand(OperandValue::Pair(a_val, b_val))
                } else {
                    // Aggregate: copy the bytes into the result place. A non-scalar volatile load is
                    // only emitted for a place destination, so `result_place` is always `Some` here.
                    let dest = result_place
                        .expect("volatile_load of a non-scalar type requires a result place");
                    let size = self.const_usize(result_layout.size.bytes());
                    self.memcpy(dest.llval, dest.align, ptr, Align::ONE, size, MemFlags::empty(), None);
                    IntrinsicResult::WroteIntoPlace
                }
            }
            // `catch_unwind(try, data, catch)`: this backend aborts on panic (it emits no unwind
            // tables), so the catch path is never reached. Run the try function and report that no
            // panic was caught (return 0) — exactly the panic=abort lowering.
            sym::catch_unwind => {
                // Pure FRAME-mode unwind tables can't run a personality, so the catch path is never
                // reached: run try(data) and report no catch (return 0), matching panic=abort.
                let try_func = args[0].immediate();
                let data = args[1].immediate();
                let i32t = self.cx.intern_type(TypeData::Int(32));
                self.emit_call_core(None, None, try_func, &[data]);
                let res = self.alloc_slot(4, 4);
                self.load_imm(X9, 0, OperandSize::S32);
                self.emit_mem_gpr(false, false, MemSize::W, X9, SP, res);
                IntrinsicResult::Operand(OperandValue::Immediate(Value::Slot { off: res, ty: i32t }))
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
            sym::sqrtf16 | sym::sqrtf32 | sym::sqrtf64 => {
                let r = self.fp_unary(FpOp1::Fsqrt, args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::floorf16 | sym::floorf32 | sym::floorf64 => {
                let r = self.fp_unary(FpOp1::Frintm, args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::ceilf16 | sym::ceilf32 | sym::ceilf64 => {
                let r = self.fp_unary(FpOp1::Frintp, args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::truncf16 | sym::truncf32 | sym::truncf64 => {
                let r = self.fp_unary(FpOp1::Frintz, args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            // `round` is ties-away-from-zero (`frinta`); `round_ties_even` is ties-to-even (`frintn`).
            sym::roundf16 | sym::roundf32 | sym::roundf64 => {
                let r = self.fp_unary(FpOp1::Frinta, args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::round_ties_even_f16 | sym::round_ties_even_f32 | sym::round_ties_even_f64 => {
                let r = self.fp_unary(FpOp1::Frintn, args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            // `fabs` is generic over the float type; f32/f64 map to `fabs`, wider/narrower floats
            // are unsupported and fall through to the loud "must be overridden" error.
            sym::fabs => {
                let arg = args[0].immediate();
                match self.cx.type_data(arg.ty()) {
                    // `f16`/`f32`/`f64` have a native `fabs` (the half form needs FEAT_FP16).
                    TypeData::Float(16) | TypeData::Float(32) | TypeData::Float(64) => {
                        let r = self.fp_unary(FpOp1::Fabs, arg);
                        IntrinsicResult::Operand(OperandValue::Immediate(r))
                    }
                    TypeData::Float(128) => {
                        let r = self.f128_fabs(arg);
                        IntrinsicResult::Operand(OperandValue::Immediate(r))
                    }
                    _ => IntrinsicResult::Fallback(instance),
                }
            }
            // Fused multiply-add (`fma`, single rounding) and `fmuladd` (fusing permitted) -> `fmadd`.
            sym::fmaf16
            | sym::fmaf32
            | sym::fmaf64
            | sym::fmuladdf16
            | sym::fmuladdf32
            | sym::fmuladdf64 => {
                let r =
                    self.fp_fma(args[0].immediate(), args[1].immediate(), args[2].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            // The transcendental routines and `copysign`/`pow` have no exact single-instruction
            // form, so they call the corresponding libm routine (always linked in `libSystem`).
            sym::sinf16 | sym::sinf32 | sym::sinf64 => {
                let r = self.fp_libm_unary("sin", args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::cosf16 | sym::cosf32 | sym::cosf64 => {
                let r = self.fp_libm_unary("cos", args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::expf16 | sym::expf32 | sym::expf64 => {
                let r = self.fp_libm_unary("exp", args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::exp2f16 | sym::exp2f32 | sym::exp2f64 => {
                let r = self.fp_libm_unary("exp2", args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::logf16 | sym::logf32 | sym::logf64 => {
                let r = self.fp_libm_unary("log", args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::log2f16 | sym::log2f32 | sym::log2f64 => {
                let r = self.fp_libm_unary("log2", args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::log10f16 | sym::log10f32 | sym::log10f64 => {
                let r = self.fp_libm_unary("log10", args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::powf16 | sym::powf32 | sym::powf64 => {
                let r = self.fp_libm_binary("pow", args[0].immediate(), args[1].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::copysignf16 | sym::copysignf32 | sym::copysignf64 => {
                let r = self.fp_libm_binary("copysign", args[0].immediate(), args[1].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::copysignf128 => {
                // No `f128` libm on macOS; combine the sign/magnitude on the GPR words inline.
                let r = self.f128_copysign(args[0].immediate(), args[1].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            // Integer power -> the compiler-builtins routine `__powidf2`/`__powisf2`.
            sym::powif16 | sym::powif32 | sym::powif64 => {
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
            sym::simd_add | sym::simd_sub | sym::simd_mul | sym::simd_div | sym::simd_rem => {
                let kind = match name {
                    sym::simd_add => SimdArith::Add,
                    sym::simd_sub => SimdArith::Sub,
                    sym::simd_mul => SimdArith::Mul,
                    sym::simd_div => SimdArith::Div,
                    _ => SimdArith::Rem,
                };
                let signed = args[0].layout.ty.simd_size_and_type(self.cx.tcx).1.is_signed();
                let r = self.emit_simd_arith(args[0].immediate(), args[1].immediate(), kind, signed);
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::simd_saturating_add | sym::simd_saturating_sub => {
                let is_add = name == sym::simd_saturating_add;
                let signed = args[0].layout.ty.simd_size_and_type(self.cx.tcx).1.is_signed();
                let r = self.emit_simd_saturating(
                    args[0].immediate(),
                    args[1].immediate(),
                    is_add,
                    signed,
                );
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::simd_ctpop | sym::simd_ctlz | sym::simd_cttz | sym::simd_bswap
            | sym::simd_bitreverse => {
                let op = match name {
                    sym::simd_ctpop => SimdBitOp::Ctpop,
                    sym::simd_ctlz => SimdBitOp::Ctlz,
                    sym::simd_cttz => SimdBitOp::Cttz,
                    sym::simd_bswap => SimdBitOp::Bswap,
                    _ => SimdBitOp::Bitreverse,
                };
                let r = self.emit_simd_bit_unary(op, args[0].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::simd_fabs | sym::simd_fsqrt | sym::simd_ceil | sym::simd_floor | sym::simd_round
            | sym::simd_trunc | sym::simd_round_ties_even => {
                let op = match name {
                    sym::simd_fabs => crate::mach::inst::FpOp1::Fabs,
                    sym::simd_fsqrt => crate::mach::inst::FpOp1::Fsqrt,
                    sym::simd_ceil => crate::mach::inst::FpOp1::Frintp,
                    sym::simd_floor => crate::mach::inst::FpOp1::Frintm,
                    sym::simd_trunc => crate::mach::inst::FpOp1::Frintz,
                    sym::simd_round_ties_even => crate::mach::inst::FpOp1::Frintn,
                    _ => crate::mach::inst::FpOp1::Frinta, // simd_round (ties away)
                };
                let r = self.emit_simd_fp_unary(args[0].immediate(), op);
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::simd_fma | sym::simd_relaxed_fma => {
                let r = self.emit_simd_fma(args[0].immediate(), args[1].immediate(), args[2].immediate());
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::simd_neg => {
                let a = args[0].immediate();
                let (elem, count, es) = self.vector_info(a.ty());
                let r = if type_is_float(self.cx, elem) {
                    self.emit_simd_fp_unary(a, crate::mach::inst::FpOp1::Fneg)
                } else if let Some((q, isize)) = self.simd_native(a.ty()) {
                    // Native-width integer vectors negate in a single `neg` instruction.
                    self.emit_simd_two(SimdUnOp::Neg, q, isize, a, a.ty())
                } else if let Some(aoff) = self.vector_to_slot(a) {
                    let (size, align) = self.cx.type_size_align(a.ty());
                    let roff = self.alloc_slot(size, align);
                    let msize = mem_size(self.cx, elem);
                    for i in 0..count {
                        self.emit_mem_gpr(true, false, msize, X10, SP, aoff + i * es);
                        self.emit(Inst::AddSubReg { op: AddSub::Sub, size: OperandSize::S64, set_flags: false, rd: X10, rn: ZR, rm: X10, amount: 0 });
                        self.emit_mem_gpr(false, false, msize, X10, SP, roff + i * es);
                    }
                    Value::Slot { off: roff, ty: a.ty() }
                } else {
                    Value::Undef { ty: a.ty() }
                };
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::simd_shl | sym::simd_shr => {
                // The right shift is arithmetic for signed lanes, logical for unsigned.
                let signed = args[0].layout.ty.simd_size_and_type(self.cx.tcx).1.is_signed();
                let left = name == sym::simd_shl;
                let r = self.emit_simd_shift(args[0].immediate(), args[1].immediate(), left, signed);
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::simd_cast | sym::simd_as => {
                // Lane-wise numeric cast (`as` per lane), across integer and floating-point lanes.
                let src_signed = args[0].layout.ty.simd_size_and_type(self.cx.tcx).1.is_signed();
                let dst_signed = result_layout.ty.simd_size_and_type(self.cx.tcx).1.is_signed();
                let dst_ty = self.cx.immediate_backend_type(result_layout);
                let r = self.emit_simd_cast(args[0].immediate(), src_signed, dst_signed, dst_ty);
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::simd_bitmask => {
                let result_ty = self.cx.immediate_backend_type(result_layout);
                let r = self.emit_simd_bitmask(args[0].immediate(), result_ty);
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::simd_reduce_add_ordered | sym::simd_reduce_mul_ordered => {
                // The ordered reductions carry an explicit accumulator (`args[1]`) and fold the
                // lanes into it left-to-right, preserving the floating-point evaluation order.
                let op = if name == sym::simd_reduce_mul_ordered {
                    SimdReduce::Mul
                } else {
                    SimdReduce::Add
                };
                let signed = args[0].layout.ty.simd_size_and_type(self.cx.tcx).1.is_signed();
                let result_ty = self.cx.immediate_backend_type(result_layout);
                let r = self.emit_simd_reduce(
                    op,
                    args[0].immediate(),
                    Some(args[1].immediate()),
                    signed,
                    result_ty,
                );
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::simd_reduce_max | sym::simd_reduce_min => {
                let op = if name == sym::simd_reduce_max { SimdReduce::Max } else { SimdReduce::Min };
                let signed = args[0].layout.ty.simd_size_and_type(self.cx.tcx).1.is_signed();
                let result_ty = self.cx.immediate_backend_type(result_layout);
                let r = self.emit_simd_reduce(op, args[0].immediate(), None, signed, result_ty);
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            sym::simd_reduce_and | sym::simd_reduce_or | sym::simd_reduce_xor => {
                let op = match name {
                    sym::simd_reduce_and => SimdReduce::And,
                    sym::simd_reduce_or => SimdReduce::Or,
                    _ => SimdReduce::Xor,
                };
                let result_ty = self.cx.immediate_backend_type(result_layout);
                let r = self.emit_simd_reduce(op, args[0].immediate(), None, false, result_ty);
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
            sym::simd_select => {
                let result_ty = self.cx.immediate_backend_type(result_layout);
                let r = self.emit_simd_select(
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
            sym::simd_insert => {
                let idx = match args[1].immediate() {
                    Value::Const { bits, .. } => bits as u64,
                    _ => 0,
                };
                let result_ty = self.cx.immediate_backend_type(result_layout);
                let r = self.emit_simd_insert(
                    args[0].immediate(),
                    idx,
                    args[2].immediate(),
                    result_ty,
                );
                IntrinsicResult::Operand(OperandValue::Immediate(r))
            }
            // `is_val_statically_known(_)` is an optimization hint with no MIR body. This backend
            // performs no such analysis, so it conservatively answers `false`; the only effect is
            // that callers take their runtime (non-constant) path, which is always correct.
            sym::is_val_statically_known => {
                let ty = self.cx.immediate_backend_type(result_layout);
                IntrinsicResult::Operand(OperandValue::Immediate(self.cx.const_int(ty, 0)))
            }
            // `select_unpredictable(cond, t, f)` -> `if cond { t } else { f }`, just a branchless
            // select (the "unpredictable" hint is ignored). Scalars/pairs pick each register; an
            // aggregate selects between the two source pointers and copies into the result place.
            sym::select_unpredictable => {
                let cond = args[0].immediate();
                match (args[1].val, args[2].val) {
                    (OperandValue::Immediate(t), OperandValue::Immediate(f)) => {
                        IntrinsicResult::Operand(OperandValue::Immediate(self.select(cond, t, f)))
                    }
                    (OperandValue::Pair(t0, t1), OperandValue::Pair(f0, f1)) => {
                        let a = self.select(cond, t0, f0);
                        let b = self.select(cond, t1, f1);
                        IntrinsicResult::Operand(OperandValue::Pair(a, b))
                    }
                    (OperandValue::Ref(t), OperandValue::Ref(f)) => {
                        let dest = result_place.expect("select_unpredictable aggregate needs a place");
                        let src = self.select(cond, t.llval, f.llval);
                        let size = self.const_usize(result_layout.size.bytes());
                        self.memcpy(dest.llval, dest.align, src, t.align.min(f.align), size, MemFlags::empty(), None);
                        IntrinsicResult::WroteIntoPlace
                    }
                    _ => IntrinsicResult::Operand(args[1].val),
                }
            }
            // Everything else falls back to the intrinsic's MIR body (if it has one).
            _ => IntrinsicResult::Fallback(instance),
        }
    }
    fn codegen_llvm_intrinsic_call(
        &mut self,
        instance: Instance<'tcx>,
        args: &[OperandRef<'tcx, Value>],
        _is_cleanup: bool,
    ) -> Value {
        let sym = self.cx.tcx.symbol_name(instance).name;
        // `isb` is a CPU pipeline hint (emitted by `spin_loop`); emit the real barrier.
        if sym == "llvm.aarch64.isb" {
            self.emit(Inst::Isb);
            return Value::Undef { ty: self.ptr_ty() };
        }
        // NEON vector intrinsics that the generic `simd_*` family can't express. Each operates on
        // whole vectors living in frame slots and is lowered lane-by-lane. The operation name is
        // the segment after the `llvm.aarch64.neon.` prefix and before the LLVM type suffix.
        if let Some(rest) = sym.strip_prefix("llvm.aarch64.neon.") {
            match rest.split('.').next().unwrap_or(rest) {
                // Pairwise unsigned maximum (`vpmaxq_u8`, memchr's "any match?" fast path).
                "umaxp" => return self.emit_neon_umaxp(args[0].immediate(), args[1].immediate()),
                // Byte table lookup (`vqtbl1q_u8`, aho-corasick's Teddy matcher).
                "tbl1" => return self.emit_neon_tbl1(args[0].immediate(), args[1].immediate()),
                // Pairwise add long, widening to the next lane size (`vpaddlq_*`/`vpadalq_*`).
                "uaddlp" => {
                    let ret = self.intrinsic_result_ty(instance);
                    return self.emit_neon_addlp(args[0].immediate(), false, ret);
                }
                "saddlp" => {
                    let ret = self.intrinsic_result_ty(instance);
                    return self.emit_neon_addlp(args[0].immediate(), true, ret);
                }
                // Pairwise add, same lane width (`vpadd_*`).
                "addp" => {
                    let ret = self.intrinsic_result_ty(instance);
                    return self.emit_neon_addp(args[0].immediate(), args[1].immediate(), ret);
                }
                // Widening multiply, doubling the lane size (`vmull_*`).
                "umull" => {
                    let ret = self.intrinsic_result_ty(instance);
                    return self.emit_neon_mull(args[0].immediate(), args[1].immediate(), false, ret);
                }
                "smull" => {
                    let ret = self.intrinsic_result_ty(instance);
                    return self.emit_neon_mull(args[0].immediate(), args[1].immediate(), true, ret);
                }
                _ => {}
            }
        }
        // AArch64 CRC32 instructions (`llvm.aarch64.crc32{c}{b,h,w,x}`): one CRC step folding a data
        // operand (arg 1) into the 32-bit accumulator (arg 0). Used by crc32fast and similar crates.
        if let Some(step) = sym.strip_prefix("llvm.aarch64.crc32") {
            let (op, size) = match step {
                "b" => (DataProc2::Crc32b, OperandSize::S32),
                "h" => (DataProc2::Crc32h, OperandSize::S32),
                "w" => (DataProc2::Crc32w, OperandSize::S32),
                "x" => (DataProc2::Crc32x, OperandSize::S64),
                "cb" => (DataProc2::Crc32cb, OperandSize::S32),
                "ch" => (DataProc2::Crc32ch, OperandSize::S32),
                "cw" => (DataProc2::Crc32cw, OperandSize::S32),
                "cx" => (DataProc2::Crc32cx, OperandSize::S64),
                _ => todo!("rustc_codegen_arm64: codegen_llvm_intrinsic_call: {sym}"),
            };
            return self.emit_crc32(op, size, args[0].immediate(), args[1].immediate());
        }
        // ARMv8 cryptography extension (`llvm.aarch64.crypto.*`): AES rounds and the SHA-1/SHA-256
        // round/schedule functions. The vector operands are loaded into `v` registers, the real
        // crypto instruction runs, and the result is stored back to a fresh frame slot. The `rd`
        // register is read-modify-write for the AES and the three-operand SHA forms.
        if let Some(op) = sym.strip_prefix("llvm.aarch64.crypto.") {
            let ret = self.intrinsic_result_ty(instance);
            match op {
                // AES: (data, key) -> Vd = data (read-modify-write), Vn = key.
                "aese" | "aesd" => {
                    let cop = if op == "aese" { CryptoTwoOp::Aese } else { CryptoTwoOp::Aesd };
                    self.load_q(args[0].immediate(), V16);
                    self.load_q(args[1].immediate(), V17);
                    self.emit(Inst::CryptoTwo { op: cop, rd: V16, rn: V17 });
                    return self.store_q(V16, ret);
                }
                // MixColumns: (data) -> Vn = data, Vd = fresh output.
                "aesmc" | "aesimc" => {
                    let cop = if op == "aesmc" { CryptoTwoOp::Aesmc } else { CryptoTwoOp::Aesimc };
                    self.load_q(args[0].immediate(), V17);
                    self.emit(Inst::CryptoTwo { op: cop, rd: V16, rn: V17 });
                    return self.store_q(V16, ret);
                }
                // SHA-256 hash update: (Qd, Qn, Vm) all v4i32.
                "sha256h" | "sha256h2" => {
                    let cop =
                        if op == "sha256h" { CryptoThreeOp::Sha256h } else { CryptoThreeOp::Sha256h2 };
                    self.load_q(args[0].immediate(), V16);
                    self.load_q(args[1].immediate(), V17);
                    self.load_q(args[2].immediate(), V18);
                    self.emit(Inst::CryptoThree { op: cop, rd: V16, rn: V17, rm: V18 });
                    return self.store_q(V16, ret);
                }
                "sha256su0" => {
                    self.load_q(args[0].immediate(), V16);
                    self.load_q(args[1].immediate(), V17);
                    self.emit(Inst::CryptoTwo { op: CryptoTwoOp::Sha256su0, rd: V16, rn: V17 });
                    return self.store_q(V16, ret);
                }
                "sha256su1" => {
                    self.load_q(args[0].immediate(), V16);
                    self.load_q(args[1].immediate(), V17);
                    self.load_q(args[2].immediate(), V18);
                    self.emit(Inst::CryptoThree { op: CryptoThreeOp::Sha256su1, rd: V16, rn: V17, rm: V18 });
                    return self.store_q(V16, ret);
                }
                // SHA-1: c/p/m take (abcd: v4i32, e: i32 scalar in an `s` reg, wk: v4i32).
                "sha1c" | "sha1p" | "sha1m" => {
                    let cop = match op {
                        "sha1c" => CryptoThreeOp::Sha1c,
                        "sha1p" => CryptoThreeOp::Sha1p,
                        _ => CryptoThreeOp::Sha1m,
                    };
                    self.load_q(args[0].immediate(), V16);
                    self.load_s32(args[1].immediate(), V17);
                    self.load_q(args[2].immediate(), V18);
                    self.emit(Inst::CryptoThree { op: cop, rd: V16, rn: V17, rm: V18 });
                    return self.store_q(V16, ret);
                }
                // SHA-1 fixed rotate: (e: i32) -> i32.
                "sha1h" => {
                    self.load_s32(args[0].immediate(), V17);
                    self.emit(Inst::CryptoTwo { op: CryptoTwoOp::Sha1h, rd: V16, rn: V17 });
                    return self.store_s32(V16, ret);
                }
                "sha1su0" => {
                    self.load_q(args[0].immediate(), V16);
                    self.load_q(args[1].immediate(), V17);
                    self.load_q(args[2].immediate(), V18);
                    self.emit(Inst::CryptoThree { op: CryptoThreeOp::Sha1su0, rd: V16, rn: V17, rm: V18 });
                    return self.store_q(V16, ret);
                }
                "sha1su1" => {
                    self.load_q(args[0].immediate(), V16);
                    self.load_q(args[1].immediate(), V17);
                    self.emit(Inst::CryptoTwo { op: CryptoTwoOp::Sha1su1, rd: V16, rn: V17 });
                    return self.store_q(V16, ret);
                }
                _ => {}
            }
        }
        todo!("rustc_codegen_arm64: codegen_llvm_intrinsic_call: {sym}")
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

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Store an inline-asm input `val` into the marshalling frame at `[sp, #off]`, in its register
    /// class's slot (a general-purpose register, or a SIMD&FP register for floats).
    fn marshal_store(&mut self, val: Value, off: u64) {
        match self.cx.type_data(val.ty()) {
            TypeData::Float(_) => {
                let sz = fp_size(self.cx, val.ty());
                self.materialize_fp(val, V0);
                self.emit_mem_fp(false, sz, V0, SP, off);
            }
            TypeData::Int(bits) if bits <= 64 => {
                self.materialize(val, X9);
                self.emit_mem_gpr(false, false, MemSize::X, X9, SP, off);
            }
            TypeData::Ptr => {
                self.materialize(val, X9);
                self.emit_mem_gpr(false, false, MemSize::X, X9, SP, off);
            }
            other => self.cx.tcx.dcx().fatal(format!(
                "rustc_codegen_arm64: unsupported inline-asm operand type {other:?} \
                 (only integers, pointers, and scalar floats are supported)"
            )),
        }
    }

    /// Load an inline-asm output from the marshalling frame at `[sp, #off]` into `place`.
    fn marshal_load(&mut self, place: PlaceRef<'tcx, Value>, off: u64) {
        let ty = self.cx.immediate_backend_type(place.layout);
        match self.cx.type_data(ty) {
            TypeData::Float(_) => {
                let sz = fp_size(self.cx, ty);
                self.emit_mem_fp(true, sz, V0, SP, off);
                let v = self.spill_fp(V0, ty);
                OperandValue::Immediate(v).store(self, place);
            }
            TypeData::Int(bits) if bits <= 64 => {
                self.emit_mem_gpr(true, false, MemSize::X, X9, SP, off);
                let v = self.spill(X9, ty);
                OperandValue::Immediate(v).store(self, place);
            }
            TypeData::Ptr => {
                self.emit_mem_gpr(true, false, MemSize::X, X9, SP, off);
                let v = self.spill(X9, ty);
                OperandValue::Immediate(v).store(self, place);
            }
            other => self.cx.tcx.dcx().fatal(format!(
                "rustc_codegen_arm64: unsupported inline-asm output type {other:?} \
                 (only integers, pointers, and scalar floats are supported)"
            )),
        }
    }
}

impl<'a, 'tcx> AsmBuilderMethods<'tcx> for Builder<'a, 'tcx> {
    fn codegen_inline_asm(
        &mut self,
        template: &[InlineAsmTemplatePiece],
        operands: &[InlineAsmOperandRef<'tcx, Self>],
        options: InlineAsmOptions,
        line_spans: &[Span],
        instance: Instance<'_>,
        dest: Option<BasicBlock>,
        _catch_funclet: Option<(BasicBlock, Option<&()>)>,
    ) {
        use crate::inline_asm::AsmOperand;

        let tcx = self.cx.tcx;
        // Lower operands into a descriptor list for the wrapper generator plus a parallel list of
        // the SSA values to marshal (input value, output place) at the call site.
        let mut gen_ops = Vec::with_capacity(operands.len());
        let mut marshal: Vec<(Option<Value>, Option<PlaceRef<'tcx, Value>>)> =
            Vec::with_capacity(operands.len());
        for op in operands {
            match op {
                InlineAsmOperandRef::In { reg, value } => {
                    gen_ops.push(AsmOperand::In { reg: *reg });
                    marshal.push((Some(value.immediate()), None));
                }
                InlineAsmOperandRef::Out { reg, late, place } => {
                    gen_ops.push(AsmOperand::Out { reg: *reg, late: *late, used: place.is_some() });
                    marshal.push((None, *place));
                }
                InlineAsmOperandRef::InOut { reg, late, in_value, out_place } => {
                    gen_ops.push(AsmOperand::InOut {
                        reg: *reg,
                        late: *late,
                        out_used: out_place.is_some(),
                    });
                    marshal.push((Some(in_value.immediate()), *out_place));
                }
                InlineAsmOperandRef::Const { string } => {
                    gen_ops.push(AsmOperand::Const { value: string.clone() });
                    marshal.push((None, None));
                }
                InlineAsmOperandRef::SymFn { instance: sym_instance } => {
                    let name = self.cx.mangle(tcx.symbol_name(*sym_instance).name);
                    self.cx.asm_syms.borrow_mut().insert(name.clone().into());
                    gen_ops.push(AsmOperand::Sym { value: name });
                    marshal.push((None, None));
                }
                InlineAsmOperandRef::SymStatic { def_id } => {
                    let name = self.cx.mangle(tcx.symbol_name(Instance::mono(tcx, *def_id)).name);
                    self.cx.asm_syms.borrow_mut().insert(name.clone().into());
                    gen_ops.push(AsmOperand::Sym { value: name });
                    marshal.push((None, None));
                }
                InlineAsmOperandRef::Label { .. } => {
                    let span = line_spans.first().copied().unwrap_or(rustc_span::DUMMY_SP);
                    tcx.dcx().span_fatal(
                        span,
                        "rustc_codegen_arm64: `asm!` label operands (asm goto) are not supported",
                    );
                }
            }
        }

        // Generate the wrapper function and its marshalling layout, and queue the wrapper for
        // assembly. The wrapper name embeds the enclosing function's symbol (unique per
        // monomorphization) and a per-unit index, so wrappers never collide across codegen units.
        // `cur_instance` is the enclosing function (same as `instance`, but carries the `'tcx`
        // lifetime that `symbol_name` needs).
        let enclosing = self.cx.cur_instance.get().expect("inline asm outside a function body");
        let idx = self.cx.inline_asm_index.get();
        self.cx.inline_asm_index.set(idx + 1);
        let asm_name = format!("{}__inline_asm_{}", tcx.symbol_name(enclosing).name, idx);
        let (wrapper, layout) = crate::inline_asm::generate(
            tcx,
            instance.def_id(),
            &asm_name,
            template,
            &gen_ops,
            options,
        );
        self.cx.global_asm.borrow_mut().push_str(&wrapper);

        // Marshalling frame: store inputs, call the wrapper with its address in x0, read outputs.
        let frame = self.alloc_slot(layout.slot_size.max(8), 16);
        for (i, (in_val, _)) in marshal.iter().enumerate() {
            if let (Some(val), Some(off)) = (in_val, layout.in_slots[i]) {
                self.marshal_store(*val, frame + off);
            }
        }
        self.emit_frame_addr(X0, frame);
        self.emit(Inst::Bl { sym: SymRef::new(self.cx.mangle(&asm_name)) });
        for (i, (_, place)) in marshal.iter().enumerate() {
            if let (Some(place), Some(off)) = (place, layout.out_slots[i]) {
                self.marshal_load(*place, frame + off);
            }
        }

        // Control flow out of the asm is normally emitted by the SSA layer *after* this returns
        // (a branch to the destination block, or `unreachable()` for `options(noreturn)`). The
        // only exception is when it hands us a normal-return target directly — the asm-goto and
        // may-unwind paths — which we must branch to ourselves.
        if let Some(dest) = dest {
            self.br(dest);
        }
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
            let mut fb = FunctionBuild::new(name, is_global, outgoing as u64);
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
        // A multi-register `PassMode::Cast` return (a small aggregate split across `x0:x1`, or an
        // HFA across `v0..v3`) must be scattered back into its result registers. The return ABI is
        // taken from the current function's `FnAbi`; a 1-register cast falls through to the scalar
        // handling below (the value is already an immediate of the cast register's type).
        if let Some(instance) = self.cx.cur_instance.get() {
            let fn_abi = self.cx.fn_abi_of_instance(instance, ty::List::empty());
            if let PassMode::Cast { ref cast, .. } = fn_abi.ret.mode {
                let regs = cast_regs(cast);
                if regs.len() > 1 {
                    self.ret_cast_multi(v, &regs);
                    self.emit(Inst::Ret { rn: LR });
                    return;
                }
            }
        }
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
        } else if self.is_f128(v.ty()) {
            self.materialize_q(v, V0);
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
        if self.is_int128(v.ty()) {
            // Match on a 128-bit value: a case matches iff both 64-bit words are equal. Compute
            // (lo==case.lo) & (hi==case.hi) into a register and branch on non-zero.
            self.materialize128(v, X9, X10);
            for (case, bb) in cases {
                self.load_imm(X11_HACK, case as u64 as u128, OperandSize::S64);
                self.load_imm(X12, (case >> 64) as u64 as u128, OperandSize::S64);
                self.emit(Inst::AddSubReg { op: AddSub::Sub, size: OperandSize::S64, set_flags: true, rd: ZR, rn: X9, rm: X11_HACK, amount: 0 });
                self.emit(Inst::CondSel { op: CondSel::Csinc, size: OperandSize::S64, rd: X13, rn: ZR, rm: ZR, cond: Cond::Ne }); // X13 = (lo==case.lo)
                self.emit(Inst::AddSubReg { op: AddSub::Sub, size: OperandSize::S64, set_flags: true, rd: ZR, rn: X10, rm: X12, amount: 0 });
                self.emit(Inst::CondSel { op: CondSel::Csinc, size: OperandSize::S64, rd: X16, rn: ZR, rm: ZR, cond: Cond::Ne }); // X16 = (hi==case.hi)
                self.emit(Inst::Logical { op: LogicOp::And, size: OperandSize::S64, rd: X13, rn: X13, rm: X16, amount: 0 });
                self.emit(Inst::CbNz { size: OperandSize::S64, rt: X13, target: bb.0 as Label, nonzero: true });
            }
            self.emit(Inst::B { target: else_llbb.0 as Label });
            return;
        }
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
        llty: Type,
        _fn_attrs: Option<&CodegenFnAttrs>,
        fn_abi: Option<&FnAbi<'tcx, Ty<'tcx>>>,
        llfn: Value,
        args: &[Value],
        then: BasicBlock,
        catch: BasicBlock,
        _funclet: Option<&()>,
        _instance: Option<Instance<'tcx>>,
    ) -> Value {
        // Bracket the call in a labeled range and record a cleanup landing pad: if the call
        // unwinds, the personality transfers control to the `catch` block (action 0 = cleanup).
        let begin = self.fresh_cs_label();
        let end = self.fresh_cs_label();
        let ret_fallback = match self.cx.type_data(llty) {
            TypeData::Func { ret, .. } => Some(ret),
            _ => None,
        };
        self.emit(Inst::Label(begin));
        let ret = self.emit_call_core(fn_abi, ret_fallback, llfn, args);
        self.emit(Inst::Label(end));
        self.record_call_site(begin, end, catch.0, 0);
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
        if self.is_f128(ty) {
            return self.f128_fneg(v);
        }
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
    fn frem(&mut self, lhs: Value, rhs: Value) -> Value {
        // `frem` has no hardware instruction; lower to the C library `fmod`/`fmodf`.
        self.fp_libm_binary("fmod", lhs, rhs)
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
        if self.is_int128(val_ty) {
            if let OverflowOp::Mul = oop {
                return self.checked128_mul(signed, lhs, rhs);
            }
            return self.checked128_addsub(oop, signed, lhs, rhs);
        }
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
            TypeData::Float(128) => {
                // `f128` is a 16-byte value held in a `q` register; copy its full image into a slot.
                let (size, align) = self.cx.type_size_align(ty);
                let off = self.alloc_slot(size, align);
                self.materialize(ptr, X9);
                self.copy_ptr_to_slot(off, X9, size);
                return Value::Slot { off, ty };
            }
            TypeData::Aggregate { .. } => {
                // A `>2`-register cast aggregate (e.g. a 3- or 4-element HFA) does not fit in a
                // single register; copy its full byte image into a fresh slot so the caller's
                // argument/return marshalling can split it across the right registers.
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
        cg_elem: OperandRef<'tcx, Value>,
        count: u64,
        dest: PlaceRef<'tcx, Value>,
    ) {
        // Initialize `dest` (an array place) with `count` copies of `cg_elem`. The generic codegen
        // only calls this when the element is not a memset-able byte. Emit a runtime loop that walks
        // a pointer from the first element to one past the last, storing the element at each step.
        // Unlike the LLVM backend this carries the induction pointer in a frame slot (this backend
        // has no SSA phi); the slot is read in the loop header and updated in the body.
        if count == 0 {
            return;
        }
        let elem_ty = self.cx.backend_type(cg_elem.layout);
        let ptr_ty = self.ptr_ty();
        let align = dest.val.align;
        let zero = self.const_usize(0);
        let count_v = self.const_usize(count);
        let start = dest.project_index(self, zero).val.llval;
        let end = dest.project_index(self, count_v).val.llval;

        // Induction pointer and loop bound, each held in a fixed 8-byte frame slot.
        let cur_off = self.alloc_slot(8, 8);
        self.materialize(start, X9);
        self.emit_mem_gpr(false, false, MemSize::X, X9, SP, cur_off);
        let end_off = self.alloc_slot(8, 8);
        self.materialize(end, X9);
        self.emit_mem_gpr(false, false, MemSize::X, X9, SP, end_off);

        let header = self.append_sibling_block("repeat_header");
        let body = self.append_sibling_block("repeat_body");
        let next = self.append_sibling_block("repeat_next");
        self.br(header);

        self.switch_to_block(header);
        let cur = Value::Slot { off: cur_off, ty: ptr_ty };
        let end_v = Value::Slot { off: end_off, ty: ptr_ty };
        let keep_going = self.icmp(IntPredicate::IntNE, cur, end_v);
        self.cond_br(keep_going, body, next);

        self.switch_to_block(body);
        let cur = Value::Slot { off: cur_off, ty: ptr_ty };
        cg_elem.val.store(self, PlaceRef::new_sized_aligned(cur, cg_elem.layout, align));
        let one = self.const_usize(1);
        let advanced = self.gep(elem_ty, cur, &[one]);
        self.materialize(advanced, X9);
        self.emit_mem_gpr(false, false, MemSize::X, X9, SP, cur_off);
        self.br(header);

        self.switch_to_block(next);
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
        if let TypeData::Aggregate { .. } = self.cx.type_data(val.ty()) {
            // An opaque aggregate (e.g. a multi-register cast value) is a byte image in a slot;
            // copy it out in full rather than truncating to a single register store.
            if let Value::Slot { off, .. } = val {
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
        if self.is_f128(val.ty()) {
            // `f128` is a 16-byte value stored through a `q` register. Materialize the value first:
            // for a constant/undef/symbol operand `materialize_q` uses x9 as scratch, so the pointer
            // must be loaded into x9 *after* it, or the store address would be clobbered.
            self.materialize_q(val, V0);
            self.materialize(ptr, X9);
            self.emit(Inst::LoadStoreQ { load: false, rt: V0, rn: X9, offset: 0 });
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
        if self.is_f128(val.ty()) {
            return self.f128_to_int(false, val, dest_ty);
        }
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
        if self.is_f128(val.ty()) {
            return self.f128_to_int(true, val, dest_ty);
        }
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
        if self.is_f128(dest_ty) {
            return self.int_to_f128(false, val, dest_ty);
        }
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
        if self.is_f128(dest_ty) {
            return self.int_to_f128(true, val, dest_ty);
        }
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
        // `f128` -> narrower float: a `compiler_builtins` libcall (source in `q0`, result in the
        // narrower FP register).
        if self.is_f128(val.ty()) {
            let sym = match fp_size(self.cx, dest_ty) {
                FpSize::S16 => "___trunctfhf2",
                FpSize::S32 => "___trunctfsf2",
                FpSize::S64 => "___trunctfdf2",
            };
            self.materialize_q(val, V0);
            self.emit(Inst::Bl { sym: SymRef::new(sym) });
            return self.spill_fp(V0, dest_ty);
        }
        let from = fp_size(self.cx, val.ty());
        let to = fp_size(self.cx, dest_ty);
        self.materialize_fp(val, V16);
        self.emit(Inst::FpCvt { from, to, rd: V16, rn: V16 });
        self.spill_fp(V16, dest_ty)
    }
    fn fpext(&mut self, val: Value, dest_ty: Type) -> Value {
        // narrower float -> `f128`: a `compiler_builtins` libcall (source in the narrower FP
        // register, result in `q0`).
        if self.is_f128(dest_ty) {
            let sym = match fp_size(self.cx, val.ty()) {
                FpSize::S16 => "___extendhftf2",
                FpSize::S32 => "___extendsftf2",
                FpSize::S64 => "___extenddftf2",
            };
            self.materialize_fp(val, V0);
            self.emit(Inst::Bl { sym: SymRef::new(sym) });
            return self.spill_q_val(V0, dest_ty);
        }
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
        let bool_ty = self.cx.intern_type(TypeData::Int(1));
        // The constant predicates need no comparison at all. (Neither these nor the two-condition
        // `ONE`/`UEQ` below are produced by Rust's own float lowering — `bin_op_to_fcmp_predicate`
        // only emits OEQ/UNE/OLT/OLE/OGT/OGE — but `fcmp` is implemented totally for any predicate.)
        match op {
            RealPredicate::RealPredicateFalse => return self.cx.const_bool(false),
            RealPredicate::RealPredicateTrue => return self.cx.const_bool(true),
            _ => {}
        }
        // `f128` has no hardware compare: each predicate is a `compiler_builtins` libcall returning
        // an `i32` whose sign relationship to `0` gives the boolean. Rust only emits the six
        // ordered/unordered forms below.
        if self.is_f128(lhs.ty()) {
            let (sym, cond) = match op {
                RealPredicate::RealOEQ => ("___eqtf2", Cond::Eq),
                RealPredicate::RealUNE => ("___netf2", Cond::Ne),
                RealPredicate::RealOLT => ("___lttf2", Cond::Lt),
                RealPredicate::RealOLE => ("___letf2", Cond::Le),
                RealPredicate::RealOGT => ("___gttf2", Cond::Gt),
                RealPredicate::RealOGE => ("___getf2", Cond::Ge),
                _ => todo!("rustc_codegen_arm64: f128 float predicate {op:?}"),
            };
            return self.f128_cmp(sym, cond, lhs, rhs);
        }
        let size = fp_size(self.cx, lhs.ty());
        self.materialize_fp(lhs, V16);
        self.materialize_fp(rhs, V17);
        self.emit(Inst::FpCmp { size, rn: V16, rm: V17 });
        // `ONE` (ordered and not equal) and `UEQ` (unordered or equal) are not single AArch64
        // condition codes: each is the OR of two `cset`s taken from one `fcmp`.
        if let RealPredicate::RealONE | RealPredicate::RealUEQ = op {
            let (c0, c1) = match op {
                RealPredicate::RealONE => (Cond::Mi, Cond::Gt), // a < b   ||  a > b
                _ => (Cond::Eq, Cond::Vs),                      // a == b  ||  unordered (NaN)
            };
            self.emit(Inst::CondSel {
                op: CondSel::Csinc,
                size: OperandSize::S32,
                rd: X9,
                rn: ZR,
                rm: ZR,
                cond: c0.invert(),
            });
            self.emit(Inst::CondSel {
                op: CondSel::Csinc,
                size: OperandSize::S32,
                rd: X10,
                rn: ZR,
                rm: ZR,
                cond: c1.invert(),
            });
            self.emit(Inst::Logical {
                op: LogicOp::Orr,
                size: OperandSize::S32,
                rd: X9,
                rn: X9,
                rm: X10,
                amount: 0,
            });
            return self.spill(X9, bool_ty);
        }
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
        self.spill(X9, bool_ty)
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

    fn vscale(&mut self, ty: Type) -> Value {
        // Scalable vectors (SVE) are unsupported; a fixed-length vector has scale 1, so even the
        // (never-taken) scalable-vector `memcpy` size would come out correct rather than ICE.
        self.cx.const_int(ty, 1)
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
    fn extract_element(&mut self, vec: Value, idx: Value) -> Value {
        let (elem, _count, es) = self.vector_info(vec.ty());
        // A constant index reuses the tested constant-lane extract.
        if let Value::Const { bits, .. } = idx {
            return self.emit_simd_extract(vec, bits as u64, elem);
        }
        let Some(voff) = self.vector_to_slot(vec) else { return Value::Undef { ty: elem } };
        // Form the lane address `sp + voff + idx * es` in X12, then load the lane through it.
        self.emit_frame_addr(X12, voff);
        self.materialize(idx, X10);
        if es == 1 {
            self.emit(Inst::AddSubReg { op: AddSub::Add, size: OperandSize::S64, set_flags: false, rd: X12, rn: X12, rm: X10, amount: 0 });
        } else {
            self.load_imm(X11_HACK, es as u128, OperandSize::S64);
            self.emit(Inst::Madd { size: OperandSize::S64, rd: X12, rn: X10, rm: X11_HACK, ra: X12 });
        }
        if type_is_float(self.cx, elem) {
            self.emit_mem_fp(true, fp_size(self.cx, elem), V16, X12, 0);
            self.spill_fp(V16, elem)
        } else {
            self.emit_mem_gpr(true, false, mem_size(self.cx, elem), X10, X12, 0);
            self.spill(X10, elem)
        }
    }
    fn vector_splat(&mut self, num_elts: usize, elt: Value) -> Value {
        // Broadcast a scalar into every lane of a fresh vector (the same lowering as `simd_splat`).
        let vec_ty = self.cx.intern_type(TypeData::Vector(elt.ty(), num_elts as u64));
        self.emit_simd_splat(elt, vec_ty)
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
        // A landing pad is entered by the unwinder with the exception object pointer in x0 and the
        // selector in x1. Spill both so the cleanup code (and the eventual `resume`) can use them.
        let ptr = self.ptr_ty();
        let i32_ty = self.cx.intern_type(TypeData::Int(32));
        let exn = self.spill(X0, ptr);
        let sel = self.spill(X1, i32_ty);
        (exn, sel)
    }
    fn filter_landing_pad(&mut self, _pers_fn: Function) {}
    fn resume(&mut self, exn0: Value, _exn1: Value) {
        // Continue unwinding to the next frame: `_Unwind_Resume(exn)` (never returns).
        self.materialize(exn0, X0);
        self.emit(Inst::Bl { sym: SymRef::new("__Unwind_Resume") });
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
            // `nand` has no single LSE op; do a CAS retry loop: new = ~(old & src).
            AtomicRmwBinOp::AtomicNand => {
                let pslot = self.alloc_slot(8, 8);
                let oslot = self.alloc_slot(8, 8);
                self.materialize(dst, X9);
                self.emit_mem_gpr(false, false, MemSize::X, X9, SP, pslot);
                self.materialize(src, X12);
                self.emit_mem_gpr(true, false, size, X10, X9, 0); // old = *ptr
                self.emit_mem_gpr(false, false, MemSize::X, X10, SP, oslot);
                let head = self.append_sibling_block("nand_loop");
                let done = self.append_sibling_block("nand_done");
                self.br(head);
                self.switch_to_block(head);
                self.emit_mem_gpr(true, false, MemSize::X, X9, SP, pslot);
                self.emit_mem_gpr(true, false, size, X10, SP, oslot); // expected
                self.materialize(src, X12);
                self.emit(Inst::Logical { op: LogicOp::And, size: OperandSize::S64, rd: X11_HACK, rn: X10, rm: X12, amount: 0 });
                self.load_imm(X13, u128::MAX, OperandSize::S64);
                self.emit(Inst::Logical { op: LogicOp::Eor, size: OperandSize::S64, rd: X11_HACK, rn: X11_HACK, rm: X13, amount: 0 }); // ~(old&src)
                self.emit(Inst::AtomicCas { acquire, release, size, rs: X10, rt: X11_HACK, rn: X9 }); // rs(old)->prev
                self.emit_mem_gpr(true, false, size, X13, SP, oslot); // expected
                self.emit(Inst::AddSubReg { op: AddSub::Sub, size: OperandSize::S64, set_flags: true, rd: ZR, rn: X10, rm: X13, amount: 0 });
                self.emit_mem_gpr(false, false, MemSize::X, X10, SP, oslot); // remember prev for next round
                self.emit(Inst::BCond { cond: Cond::Eq, target: done.0 as Label });
                self.br(head);
                self.switch_to_block(done);
                return self.spill(X10, ty);
            }
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
        llty: Type,
        _caller_attrs: Option<&CodegenFnAttrs>,
        fn_abi: Option<&FnAbi<'tcx, Ty<'tcx>>>,
        fn_val: Value,
        args: &[Value],
        _funclet: Option<&()>,
        _callee_instance: Option<Instance<'tcx>>,
    ) -> Value {
        // When the caller supplies no `FnAbi` (e.g. the entry-point wrapper calling `lang_start`),
        // recover the return type from the function type so the result is still collected.
        let ret_fallback = match self.cx.type_data(llty) {
            TypeData::Func { ret, .. } => Some(ret),
            _ => None,
        };
        self.emit_call_core(fn_abi, ret_fallback, fn_val, args)
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
                let fsize = self.cx.type_size_align(fty).0;
                if matches!(fsize, 1 | 2 | 4 | 8) {
                    let sz = mem_size_from_bytes(fsize);
                    self.emit(Inst::LoadStoreUImm { load: true, signed: false, size: sz, rt: X10, rn: X9, offset: foff });
                    self.emit_mem_gpr(false, false, sz, X10, SP, base + foff);
                } else {
                    // An odd-width integer field — e.g. the `i24`/`i40`/`i48`/`i56` remainder of a
                    // cast `Pair` — would over-read/write at a single 8-byte access (`mem_size`
                    // rounds 3/5/6/7 up to `x`); copy exactly its bytes instead.
                    self.copy_base_off_to_slot(X9, foff, base + foff, fsize);
                }
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
                let fsize = self.cx.type_size_align(fty).0;
                if matches!(fsize, 1 | 2 | 4 | 8) {
                    let sz = mem_size_from_bytes(fsize);
                    self.emit_mem_gpr(true, false, sz, X10, SP, sbase + foff);
                    self.emit(Inst::LoadStoreUImm { load: false, signed: false, size: sz, rt: X10, rn: X9, offset: foff });
                } else {
                    // Odd-width integer field (cast remainder): copy exactly its bytes.
                    self.copy_slot_to_base_off(sbase + foff, X9, foff, fsize);
                }
            }
        }
    }

    /// Marshal a single scalar argument into the next register of its bank, spilling to the
    /// outgoing-argument area once the bank is exhausted. Floats use the SIMD bank (`v0..v7`),
    /// 128-bit integers take a consecutive `x` pair (or a 16-byte stack slot), everything else uses
    /// the next `x` register.
    fn marshal_scalar_arg(&mut self, arg: Value, a: &mut ArgAssign) {
        if self.is_f128(arg.ty()) {
            // `f128` is passed in a full 128-bit `q` register (or 16-byte-aligned on the stack).
            if a.nsrn < 8 {
                self.materialize_q(arg, Vreg::from_encoding(a.nsrn));
                a.nsrn += 1;
            } else {
                a.nsaa = (a.nsaa + 15) & !15;
                self.materialize_q(arg, V16);
                self.emit_q(false, V16, a.nsaa as u64);
                a.nsaa += 16;
            }
        } else if type_is_float(self.cx, arg.ty()) {
            if a.nsrn < 8 {
                self.materialize_fp(arg, Vreg::from_encoding(a.nsrn));
                a.nsrn += 1;
            } else {
                let (size, align) = self.cx.type_size_align(arg.ty());
                let off = a.stack_slot(size as u32, align as u32);
                self.materialize_fp(arg, V16);
                self.emit_mem_fp(false, fp_size(self.cx, arg.ty()), V16, SP, off as u64);
            }
        } else if self.is_int128(arg.ty()) {
            if a.ngrn <= 6 {
                let lo = Gpr::from_encoding(a.ngrn);
                let hi = Gpr::from_encoding(a.ngrn + 1);
                self.materialize128(arg, lo, hi);
                a.ngrn += 2;
            } else {
                a.ngrn = 8;
                a.nsaa = (a.nsaa + 15) & !15;
                self.materialize128(arg, X9, X10);
                self.emit_mem_gpr(false, false, MemSize::X, X9, SP, a.nsaa as u64);
                self.emit_mem_gpr(false, false, MemSize::X, X10, SP, a.nsaa as u64 + 8);
                a.nsaa += 16;
            }
        } else if a.ngrn < 8 {
            self.materialize(arg, Gpr::from_encoding(a.ngrn));
            a.ngrn += 1;
        } else {
            let (size, align) = self.cx.type_size_align(arg.ty());
            let off = a.stack_slot(size as u32, align as u32);
            self.materialize(arg, X9);
            self.emit_mem_gpr(false, false, mem_size(self.cx, arg.ty()), X9, SP, off as u64);
        }
    }

    /// Marshal a `PassMode::Cast` argument. A 1-register cast travels as a single scalar; a
    /// multi-register cast is split into its register pieces (integer GPRs for a composite, SIMD&FP
    /// registers for an HFA), all-or-nothing: if they do not all fit in their bank the aggregate is
    /// copied contiguously onto the outgoing-argument area.
    fn marshal_cast_arg(&mut self, arg: Value, cast: &CastTarget, a: &mut ArgAssign) {
        let regs = cast_regs(cast);
        let src = match arg {
            // An aggregate cast value lives in memory; marshal each register piece from its slot.
            // This includes a single-register cast: it still travels as a register-sized chunk (an
            // 8-byte stack slot when spilled), unlike a `Direct` scalar which the caller packs at its
            // natural size — so it must use the cast path here, not `marshal_scalar_arg`.
            Value::Slot { off, .. } => off,
            // A non-memory cast value (rare) is treated as a plain scalar.
            _ => return self.marshal_scalar_arg(arg, a),
        };
        let n_int = regs.iter().filter(|r| !r.fp).count() as u8;
        let n_fp = regs.iter().filter(|r| r.fp).count() as u8;
        if a.ngrn + n_int <= 8 && a.nsrn + n_fp <= 8 {
            for r in &regs {
                if r.fp {
                    let fps = if r.width == 8 { FpSize::S64 } else { FpSize::S32 };
                    self.emit_mem_fp(true, fps, Vreg::from_encoding(a.nsrn), SP, src + r.offset as u64);
                    a.nsrn += 1;
                } else {
                    self.load_agg_gpr(Gpr::from_encoding(a.ngrn), src + r.offset as u64, r.data);
                    a.ngrn += 1;
                }
            }
        } else {
            if n_int > 0 {
                a.ngrn = 8;
            }
            if n_fp > 0 {
                a.nsrn = 8;
            }
            // Mirror `push_cast_param`'s stack layout: pieces back-to-back over the cast's footprint
            // (their total width), at the aggregate's natural alignment.
            let footprint = regs.iter().map(|r| r.offset + r.width).max().unwrap_or(0);
            let align = cast.align(self.cx).bytes() as u32;
            let base = a.stack_slot(footprint, align);
            for r in &regs {
                self.copy_slot_to_base_off(
                    src + r.offset as u64,
                    SP,
                    (base + r.offset) as u64,
                    r.data as u64,
                );
            }
        }
    }

    /// Load `data` (1..=8) bytes from frame slot `src_off` into GPR `reg`, zero-extended. Natural
    /// widths use one load; an odd width (3/5/6/7) is staged through a zeroed 8-byte scratch.
    fn load_agg_gpr(&mut self, reg: Gpr, src_off: u64, data: u32) {
        match data {
            1 | 2 | 4 | 8 => {
                self.emit_mem_gpr(true, false, mem_size_from_bytes(data as u64), reg, SP, src_off)
            }
            _ => {
                let sc = self.alloc_slot(8, 8);
                self.emit_mem_gpr(false, false, MemSize::X, ZR, SP, sc);
                self.copy_slot_to_base_off(src_off, SP, sc, data as u64);
                self.emit_mem_gpr(true, false, MemSize::X, reg, SP, sc);
            }
        }
    }

    /// Store the low `data` (1..=8) bytes of GPR `reg` to frame slot `dst_off`. Mirror of
    /// [`load_agg_gpr`](Self::load_agg_gpr).
    fn store_agg_gpr(&mut self, reg: Gpr, dst_off: u64, data: u32) {
        match data {
            1 | 2 | 4 | 8 => {
                self.emit_mem_gpr(false, false, mem_size_from_bytes(data as u64), reg, SP, dst_off)
            }
            _ => {
                let sc = self.alloc_slot(8, 8);
                self.emit_mem_gpr(false, false, MemSize::X, reg, SP, sc);
                self.copy_slot_to_base_off(sc, SP, dst_off, data as u64);
            }
        }
    }

    /// Collect a `PassMode::Cast` return value out of its result registers into a fresh frame slot
    /// laid out as the aggregate, returning a `Value::Slot` over it.
    fn collect_cast_ret(&mut self, cast: &CastTarget, layout: TyAndLayout<'tcx>) -> Value {
        let regs = cast_regs(cast);
        let size = cast.size(self.cx).bytes().max(layout.size.bytes()).max(1);
        let align = cast.align(self.cx).bytes().max(layout.align.abi.bytes()).max(1);
        let slot = self.alloc_slot(size, align);
        let mut ngrn: u8 = 0;
        let mut nsrn: u8 = 0;
        for r in &regs {
            if r.fp {
                let fps = if r.width == 8 { FpSize::S64 } else { FpSize::S32 };
                self.emit_mem_fp(false, fps, Vreg::from_encoding(nsrn), SP, slot + r.offset as u64);
                nsrn += 1;
            } else {
                self.store_agg_gpr(Gpr::from_encoding(ngrn), slot + r.offset as u64, r.data);
                ngrn += 1;
            }
        }
        let ty = self.cx.intern_type(TypeData::Aggregate {
            size: layout.size.bytes(),
            align: layout.align.abi.bytes(),
        });
        Value::Slot { off: slot, ty }
    }

    /// Return a multi-register `PassMode::Cast` value: scatter the slot's bytes back into the
    /// result registers (integer GPRs for a composite, SIMD&FP registers for an HFA).
    fn ret_cast_multi(&mut self, v: Value, regs: &[CastReg]) {
        let off = match v {
            Value::Slot { off, .. } => off,
            // A multi-register cast value is always materialized in memory; fall back defensively.
            _ => return self.materialize(v, X0),
        };
        let mut ngrn: u8 = 0;
        let mut nsrn: u8 = 0;
        for r in regs {
            if r.fp {
                let fps = if r.width == 8 { FpSize::S64 } else { FpSize::S32 };
                self.emit_mem_fp(true, fps, Vreg::from_encoding(nsrn), SP, off + r.offset as u64);
                nsrn += 1;
            } else {
                self.load_agg_gpr(Gpr::from_encoding(ngrn), off + r.offset as u64, r.data);
                ngrn += 1;
            }
        }
    }

    /// Shared implementation of `call`/`invoke`: marshal arguments into the ABI registers, emit the
    /// branch-and-link (direct or indirect), and collect the return value.
    fn emit_call_core(
        &mut self,
        fn_abi: Option<&FnAbi<'tcx, Ty<'tcx>>>,
        ret_ty_fallback: Option<Type>,
        fn_val: Value,
        args: &[Value],
    ) -> Value {
        let indirect_ret = fn_abi.is_some_and(|a| a.ret.is_indirect());
        // Marshal physical arguments into the integer (`x0..x7`) and floating-point (`v0..v7`)
        // register banks; an sret pointer goes in x8. Once a bank is exhausted the remaining
        // arguments are written into the outgoing-argument area at the bottom of our frame
        // (`sp + nsaa`), which the frame-layout pre-pass sized so they never overlap local slots.
        //
        // With a concrete `fn_abi` we walk its argument modes so that a `PassMode::Cast` value (a
        // small aggregate or HFA) is split across the exact registers the callee expects; without
        // one (the entry-wrapper/`catch_unwind` paths) each value is a single scalar register.
        let mut a = ArgAssign::default();
        let mut ai = 0usize;
        if indirect_ret {
            self.materialize(args[ai], Gpr::from_encoding(8));
            ai += 1;
        }
        match fn_abi {
            Some(abi) => {
                for (argn, arg_abi) in abi.args.iter().enumerate() {
                    // Apple AArch64 passes every variadic argument (those past the last named
                    // parameter) on the stack, regardless of type; exhaust both register banks
                    // once the variadic portion begins so the marshalling spills them.
                    if abi.c_variadic && argn == abi.fixed_count as usize {
                        a.ngrn = 8;
                        a.nsrn = 8;
                    }
                    match arg_abi.mode {
                        PassMode::Ignore => {}
                        PassMode::Cast { ref cast, .. } => {
                            self.marshal_cast_arg(args[ai], cast, &mut a);
                            ai += 1;
                        }
                        PassMode::Pair(..) => {
                            self.marshal_scalar_arg(args[ai], &mut a);
                            self.marshal_scalar_arg(args[ai + 1], &mut a);
                            ai += 2;
                        }
                        PassMode::Indirect { meta_attrs, .. } => {
                            self.marshal_scalar_arg(args[ai], &mut a);
                            ai += 1;
                            if meta_attrs.is_some() {
                                self.marshal_scalar_arg(args[ai], &mut a);
                                ai += 1;
                            }
                        }
                        PassMode::Direct(_) => {
                            self.marshal_scalar_arg(args[ai], &mut a);
                            ai += 1;
                        }
                    }
                }
            }
            None => {
                for &arg in &args[ai..] {
                    self.marshal_scalar_arg(arg, &mut a);
                }
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
        // Collect the return value from x0 / v0 (scalar baseline). With a concrete `fn_abi` we
        // honor its return mode; without one — e.g. the synthesized C `main` wrapper's call to
        // `lang_start`, which passes the function type but no `FnAbi` — we fall back to the
        // callee's declared return type so the result (the process exit code!) isn't dropped.
        // A `PassMode::Cast` return (a small aggregate or HFA) comes back in one or more registers
        // determined by the cast, not by the Rust layout; collect them into a slot laid out as the
        // aggregate. `store_arg` then copies that slot into the destination place.
        if let Some(a) = fn_abi {
            if let PassMode::Cast { ref cast, .. } = a.ret.mode {
                return self.collect_cast_ret(cast, a.ret.layout);
            }
        }
        let ret_ty = match fn_abi {
            Some(a) if a.ret.is_indirect() || a.ret.is_ignore() => None,
            Some(a) => Some(self.cx.immediate_backend_type(a.ret.layout)),
            None => ret_ty_fallback.filter(|t| !matches!(self.cx.type_data(*t), TypeData::Void)),
        };
        match ret_ty {
            Some(ty) => {
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
                    _ if self.is_f128(ty) => self.spill_q_val(V0, ty),
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
