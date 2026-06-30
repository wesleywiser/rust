//! The per-codegen-unit context (`CodegenCx`) and the backend's `Copy` value/type handles.
//!
//! Types and symbols are interned to small `Copy` indices so that the `BackendTypes` associated
//! types satisfy the `Copy + Eq + Hash` bounds the `rustc_codegen_ssa` traits require.

use std::cell::{Cell, RefCell};

use rustc_abi::{HasDataLayout, TargetDataLayout};
use rustc_codegen_ssa::traits::{BackendTypes, MiscCodegenMethods};
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::ty::layout::{
    FnAbiError, FnAbiOfHelpers, FnAbiRequest, HasTyCtxt, HasTypingEnv, LayoutError, LayoutOfHelpers,
};
use rustc_middle::ty::{self, Instance, Ty, TyCtxt};
use rustc_session::Session;
use rustc_span::{Span, Symbol};
use rustc_target::spec::{HasTargetSpec, HasX86AbiOpt, Target, X86Abi};

use crate::mach::module::MachModule;

/// An interned backend type handle (also used as `BackendTypes::FunctionSignature`).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Type(u32);

/// An interned symbol name (a function or data symbol).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Sym(u32);

/// A declared function handle (index into [`CodegenCx::functions`]).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Function(pub(crate) u32);

/// A basic block within the function currently being built.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct BasicBlock(pub(crate) u32);

/// Declaration metadata for a [`Function`] handle.
pub struct FuncDecl {
    pub sym: Sym,
    pub is_global: bool,
}

/// A backend SSA value. In the baseline model these are either compile-time constants, symbol
/// addresses, or (added in the builder phase) references to per-function stack slots.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Value {
    /// A scalar immediate constant of the given backend type.
    Const { bits: u128, ty: Type },
    /// The address of a named symbol plus a byte offset (functions, statics, string literals).
    Sym { sym: Sym, offset: i64, ty: Type },
    /// A scalar value (integer, float, or pointer) materialized at the given `sp`-relative frame
    /// offset. This is how the baseline keeps every non-constant SSA value live. The offset is a
    /// `u64` because frames are not bounded to 4 GiB.
    Slot { off: u64, ty: Type },
    /// An undefined/poison value of the given type.
    Undef { ty: Type },
}

impl Value {
    /// The backend type of this value (used by `val_ty`).
    pub fn ty(self) -> Type {
        match self {
            Value::Const { ty, .. }
            | Value::Sym { ty, .. }
            | Value::Slot { ty, .. }
            | Value::Undef { ty } => ty,
        }
    }
}

/// The structural description behind an interned [`Type`].
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum TypeData {
    Void,
    /// An integer of the given bit width (1 for `i1`/`bool` in registers).
    Int(u32),
    /// A float of the given bit width (16/32/64/128).
    Float(u32),
    /// A pointer (address space is not tracked in the baseline).
    Ptr,
    Array(Type, u64),
    Vector(Type, u64),
    /// A scalar pair (e.g. `(T, bool)`, a fat pointer, an overflow result) held immediately as two
    /// fields. Stored so the builder can pack/unpack pairs for the `PassMode::Pair` ABI.
    Pair(Type, Type),
    /// Any aggregate (struct/union/scalar-pair) — only its size/align matter to the baseline, which
    /// addresses fields by byte offset.
    Aggregate { size: u64, align: u64 },
    /// A function signature.
    Func { params: Vec<Type>, ret: Type },
}

struct Interner<T: Clone + Eq + std::hash::Hash> {
    items: Vec<T>,
    dedup: FxHashMap<T, u32>,
}

/// Round `value` up to a multiple of `align` (which is treated as at least 1).
fn align_up_u64(value: u64, align: u64) -> u64 {
    let align = align.max(1);
    value.div_ceil(align) * align
}

impl<T: Clone + Eq + std::hash::Hash> Default for Interner<T> {
    fn default() -> Self {
        Interner { items: Vec::new(), dedup: FxHashMap::default() }
    }
}

impl<T: Clone + Eq + std::hash::Hash> Interner<T> {
    fn intern(&mut self, item: T) -> u32 {
        if let Some(&id) = self.dedup.get(&item) {
            return id;
        }
        let id = self.items.len() as u32;
        self.items.push(item.clone());
        self.dedup.insert(item, id);
        id
    }

    fn get(&self, id: u32) -> &T {
        &self.items[id as usize]
    }
}

/// The codegen context for one codegen unit.
pub struct CodegenCx<'tcx> {
    pub tcx: TyCtxt<'tcx>,

    types: RefCell<Interner<TypeData>>,
    symbols: RefCell<Interner<Box<str>>>,

    /// The machine code accumulated for this codegen unit.
    pub module: RefCell<MachModule>,
    /// The function currently being lowered by a `Builder`, if any.
    pub cur_fn: RefCell<Option<crate::builder::FunctionBuild>>,
    /// The instance currently being lowered (set before `define`), used for ABI/arg setup.
    pub cur_instance: Cell<Option<Instance<'tcx>>>,

    /// Cache of monomorphic instances already assigned a [`Function`] handle.
    pub instances: RefCell<FxHashMap<Instance<'tcx>, Function>>,
    /// Declaration metadata for each [`Function`] handle.
    pub functions: RefCell<Vec<FuncDecl>>,
    /// Functions already declared, keyed by final symbol name (dedups declarations).
    declared_fns: RefCell<FxHashMap<String, Function>>,
    /// Vtable cache required by `MiscCodegenMethods`.
    vtables: RefCell<FxHashMap<(Ty<'tcx>, Option<ty::ExistentialTraitRef<'tcx>>), Value>>,
    /// Symbol interned for each predefined `static` item.
    pub statics: RefCell<FxHashMap<rustc_hir::def_id::DefId, Sym>>,
    /// Foreign (dylib-imported) static symbols whose address must be loaded via the GOT.
    pub got_syms: RefCell<rustc_data_structures::fx::FxHashSet<Sym>>,
    /// Symbol emitted for each anonymous constant allocation (string literals, `&CONST`, promoted
    /// constants, vtables), keyed by allocation content so identical constants are deduplicated.
    pub static_consts: RefCell<FxHashMap<rustc_middle::mir::interpret::Allocation, Sym>>,
    eh_personality: Cell<Option<Function>>,

    local_gen_sym_counter: Cell<usize>,

    /// DWARF debug-info builder for this codegen unit. `Some` iff debug info is enabled. Taken out
    /// (via [`CodegenCx::take_debug_context`]) at the end of codegen to travel with the module.
    pub debug: Option<RefCell<crate::dwarf::DebugContext>>,

    /// Accumulated textual assembly for this codegen unit (`global_asm!` blocks and `#[naked]`
    /// function bodies). Assembled by an external assembler into a side object at emit time.
    pub global_asm: RefCell<String>,

    /// Symbol names referenced by `sym` operands in this codegen unit's `global_asm!`/`asm!`. They
    /// must be emitted with global scope so the separately-assembled asm object can resolve them.
    pub asm_syms: RefCell<rustc_data_structures::fx::FxHashSet<Box<str>>>,

    /// Counter for naming the generated wrapper function of each inline `asm!` in this unit.
    pub inline_asm_index: Cell<usize>,
}

/// The target's macOS deployment version, packed as `major << 16 | minor << 8 | patch` for the
/// Mach-O `LC_BUILD_VERSION` load command (the same encoding rustc/LLVM use).
pub(crate) fn macho_min_os(tcx: TyCtxt<'_>) -> u32 {
    let v = tcx.sess.apple_deployment_target();
    ((v.major as u32) << 16) | ((v.minor as u32) << 8) | (v.patch as u32)
}

impl<'tcx> CodegenCx<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, _cgu_name: Symbol) -> CodegenCx<'tcx> {
        let mut module = MachModule::new();
        module.macho_min_os = macho_min_os(tcx);
        CodegenCx {
            tcx,
            types: RefCell::new(Interner::default()),
            symbols: RefCell::new(Interner::default()),
            module: RefCell::new(module),
            cur_fn: RefCell::new(None),
            cur_instance: Cell::new(None),
            instances: RefCell::new(FxHashMap::default()),
            functions: RefCell::new(Vec::new()),
            declared_fns: RefCell::new(FxHashMap::default()),
            vtables: RefCell::new(FxHashMap::default()),
            statics: RefCell::new(FxHashMap::default()),
            got_syms: RefCell::new(rustc_data_structures::fx::FxHashSet::default()),
            static_consts: RefCell::new(FxHashMap::default()),
            eh_personality: Cell::new(None),
            local_gen_sym_counter: Cell::new(0),
            debug: crate::dwarf::DebugContext::new(tcx).map(RefCell::new),
            global_asm: RefCell::new(String::new()),
            asm_syms: RefCell::new(rustc_data_structures::fx::FxHashSet::default()),
            inline_asm_index: Cell::new(0),
        }
    }

    /// Intern a [`TypeData`], returning its [`Type`] handle.
    pub fn intern_type(&self, data: TypeData) -> Type {
        Type(self.types.borrow_mut().intern(data))
    }

    /// Look up the structure behind a [`Type`] (cloned out of the interner).
    pub fn type_data(&self, ty: Type) -> TypeData {
        self.types.borrow().get(ty.0).clone()
    }

    /// Intern a symbol name.
    pub fn intern_sym(&self, name: &str) -> Sym {
        Sym(self.symbols.borrow_mut().intern(name.into()))
    }

    /// The textual name behind an interned [`Sym`].
    pub fn sym_name(&self, sym: Sym) -> Box<str> {
        self.symbols.borrow().get(sym.0).clone()
    }

    /// The size (bytes) and alignment (bytes) of a backend type.
    pub fn type_size_align(&self, ty: Type) -> (u64, u64) {
        match self.type_data(ty) {
            TypeData::Int(bits) => {
                let bytes = (bits.max(8) / 8) as u64;
                (bytes, bytes)
            }
            TypeData::Float(bits) => {
                let bytes = (bits / 8) as u64;
                (bytes, bytes)
            }
            TypeData::Ptr => (8, 8),
            TypeData::Array(elem, count) => {
                let (es, ea) = self.type_size_align(elem);
                (es * count, ea)
            }
            TypeData::Vector(elem, count) => {
                let (es, _) = self.type_size_align(elem);
                (es * count, es * count)
            }
            TypeData::Pair(a, b) => {
                let (sa, aa) = self.type_size_align(a);
                let (sb, ab) = self.type_size_align(b);
                let align = aa.max(ab).max(1);
                let off1 = align_up_u64(sa, ab.max(1));
                (align_up_u64(off1 + sb, align), align)
            }
            TypeData::Aggregate { size, align } => (size, align),
            TypeData::Func { .. } | TypeData::Void => (0, 1),
        }
    }

    /// The byte offset and backend type of field `idx` (0 or 1) of a [`TypeData::Pair`].
    pub fn pair_field(&self, pair_ty: Type, idx: usize) -> (u64, Type) {
        match self.type_data(pair_ty) {
            TypeData::Pair(a, b) => {
                if idx == 0 {
                    (0, a)
                } else {
                    let (sa, _) = self.type_size_align(a);
                    let (_, ab) = self.type_size_align(b);
                    (align_up_u64(sa, ab.max(1)), b)
                }
            }
            other => panic!("pair_field on non-pair type {other:?}"),
        }
    }

    /// Generate a fresh internal symbol name with the given prefix.
    pub fn generate_local_symbol_name(&self, prefix: &str) -> String {
        let idx = self.local_gen_sym_counter.get();
        self.local_gen_sym_counter.set(idx + 1);
        format!("{prefix}.{idx}")
    }
}

impl<'tcx> BackendTypes for CodegenCx<'tcx> {
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

impl<'tcx> HasTyCtxt<'tcx> for CodegenCx<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

impl HasDataLayout for CodegenCx<'_> {
    fn data_layout(&self) -> &TargetDataLayout {
        &self.tcx.data_layout
    }
}

impl HasTargetSpec for CodegenCx<'_> {
    fn target_spec(&self) -> &Target {
        &self.tcx.sess.target
    }
}

impl HasX86AbiOpt for CodegenCx<'_> {
    fn x86_abi_opt(&self) -> X86Abi {
        X86Abi {
            regparm: self.tcx.sess.opts.unstable_opts.regparm,
            reg_struct_return: self.tcx.sess.opts.unstable_opts.reg_struct_return,
        }
    }
}

impl<'tcx> HasTypingEnv<'tcx> for CodegenCx<'tcx> {
    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        ty::TypingEnv::fully_monomorphized()
    }
}

impl<'tcx> LayoutOfHelpers<'tcx> for CodegenCx<'tcx> {
    #[inline]
    fn handle_layout_err(&self, err: LayoutError<'tcx>, span: Span, ty: Ty<'tcx>) -> ! {
        self.tcx.dcx().span_fatal(span, format!("failed to get layout for `{ty}`: {err:?}"))
    }
}

impl<'tcx> FnAbiOfHelpers<'tcx> for CodegenCx<'tcx> {
    #[inline]
    fn handle_fn_abi_err(
        &self,
        err: FnAbiError<'tcx>,
        span: Span,
        fn_abi_request: FnAbiRequest<'tcx>,
    ) -> ! {
        if let FnAbiError::Layout(LayoutError::SizeOverflow(_)) = err {
            self.tcx.dcx().span_fatal(span, format!("{err:?}"))
        } else {
            self.tcx
                .dcx()
                .span_bug(span, format!("failed to compute fn_abi ({fn_abi_request:?}): {err:?}"))
        }
    }
}

impl<'tcx> CodegenCx<'tcx> {
    /// Apply the platform symbol mangling. On Mach-O all symbols gain a leading underscore.
    pub fn mangle(&self, name: &str) -> String {
        format!("_{name}")
    }

    /// Declare a function by final symbol name, deduplicating repeated declarations.
    pub fn declare_named_fn(&self, name: &str, is_global: bool) -> Function {
        if let Some(&f) = self.declared_fns.borrow().get(name) {
            return f;
        }
        let sym = self.intern_sym(name);
        let func = {
            let mut funcs = self.functions.borrow_mut();
            let id = funcs.len() as u32;
            funcs.push(FuncDecl { sym, is_global });
            Function(id)
        };
        self.declared_fns.borrow_mut().insert(name.to_owned(), func);
        func
    }

    /// The interned symbol of a declared function.
    pub fn function_sym(&self, func: Function) -> Sym {
        self.functions.borrow()[func.0 as usize].sym
    }
}

impl<'tcx> MiscCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn vtables(
        &self,
    ) -> &RefCell<FxHashMap<(Ty<'tcx>, Option<ty::ExistentialTraitRef<'tcx>>), Value>> {
        &self.vtables
    }

    fn get_fn(&self, instance: Instance<'tcx>) -> Function {
        if let Some(&f) = self.instances.borrow().get(&instance) {
            return f;
        }
        let name = self.mangle(self.tcx.symbol_name(instance).name);
        let func = self.declare_named_fn(&name, true);
        self.instances.borrow_mut().insert(instance, func);
        func
    }

    fn get_fn_addr(&self, instance: Instance<'tcx>) -> Value {
        let func = self.get_fn(instance);
        let sym = self.function_sym(func);
        // Taking the address of a dylib-imported (foreign) function needs the GOT, same as a
        // foreign static: a direct adrp+add cannot be fixed up (`_utimes` does not have address).
        if self.tcx.is_foreign_item(instance.def_id()) {
            self.got_syms.borrow_mut().insert(sym);
        }
        Value::Sym { sym, offset: 0, ty: self.intern_type(TypeData::Ptr) }
    }

    fn eh_personality(&self) -> Function {
        if let Some(f) = self.eh_personality.get() {
            return f;
        }
        let f = self.declare_named_fn("_rust_eh_personality", true);
        self.eh_personality.set(Some(f));
        f
    }

    fn sess(&self) -> &Session {
        self.tcx.sess
    }

    fn set_frame_pointer_type(&self, _llfn: Function) {}
    fn apply_target_cpu_attr(&self, _llfn: Function) {}

    fn declare_c_main(&self, _fn_type: Type) -> Option<Function> {
        let name = "_main";
        if self.declared_fns.borrow().contains_key(name) {
            return None;
        }
        Some(self.declare_named_fn(name, true))
    }

    fn intrinsic_call_expects_place_always(&self, _name: Symbol) -> bool {
        false
    }
}
