//! Allocator shim.
//!
//! The standard library calls into a small set of `__rust_*` symbols to reach the global
//! allocator. For the default allocator each one is a thin forwarder to the corresponding
//! `__rdl_*` implementation provided by `std`; we also emit the `__rust_no_alloc_shim_is_unstable_v2`
//! marker that the library references to force the shim to be linked in.
//!
//! Each forwarder is built so the incoming argument registers (`x0..x7`) are left untouched across
//! a framed `bl`, so they reach the target unchanged and the result in `x0` survives the epilogue.
//! This means a single shape works for every method regardless of its argument count.

use rustc_ast::expand::allocator::{
    AllocatorMethod, NO_ALLOC_SHIM_IS_UNSTABLE, default_fn_name, global_fn_name,
};
use rustc_middle::ty::TyCtxt;
use rustc_symbol_mangling::mangle_internal_symbol;

use crate::builder::FunctionBuild;
use crate::context::Function;
use crate::mach::func::MachFunction;
use crate::mach::inst::{Inst, SymRef};
use crate::mach::module::MachModule;
use crate::mach::reg::LR;

/// Build the allocator shim module for the given set of allocator methods.
pub(crate) fn codegen(tcx: TyCtxt<'_>, methods: &[AllocatorMethod]) -> MachModule {
    let mut module = MachModule::new();
    for method in methods {
        let wrapper = mach_symbol(tcx, &global_fn_name(method.name));
        let target = mach_symbol(tcx, &default_fn_name(method.name));
        module.push_function(forwarder(wrapper, &target));
    }
    // `std` references this marker symbol to ensure the shim is linked; it just has to exist.
    module.push_function(empty(mach_symbol(tcx, NO_ALLOC_SHIM_IS_UNSTABLE)));
    module
}

/// The final Mach-O symbol for an internal runtime symbol: `mangle_internal_symbol` produces the
/// `_R..`-mangled name and Mach-O adds the platform leading underscore on top of that.
fn mach_symbol(tcx: TyCtxt<'_>, name: &str) -> String {
    format!("_{}", mangle_internal_symbol(tcx, name))
}

/// A forwarder `wrapper(args..) -> target(args..)` that returns the target's result. The function
/// id is irrelevant for a standalone shim function (it is unused by `FunctionBuild::finish`).
fn forwarder(name: String, target: &str) -> MachFunction {
    let mut fb = FunctionBuild::new(Function(0), name.into(), true);
    let block = fb.new_block();
    fb.blocks[block.0 as usize].push(Inst::Bl { sym: SymRef::new(target) });
    fb.blocks[block.0 as usize].push(Inst::Ret { rn: LR });
    fb.finish()
}

/// An `extern "C"` function that immediately returns.
fn empty(name: String) -> MachFunction {
    let mut fb = FunctionBuild::new(Function(0), name.into(), true);
    let block = fb.new_block();
    fb.blocks[block.0 as usize].push(Inst::Ret { rn: LR });
    fb.finish()
}
