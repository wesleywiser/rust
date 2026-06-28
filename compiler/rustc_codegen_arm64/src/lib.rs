//! `rustc_codegen_arm64` is a fast, no-optimization codegen backend for rustc that lowers MIR
//! directly to AArch64 machine code. Its only goal is *compile speed*; generated code quality is
//! explicitly a non-goal.
//!
//! It reuses `rustc_codegen_ssa`'s generic MIR-lowering driver (like the LLVM and GCC backends) by
//! implementing the backend trait family, rather than walking MIR itself.

// tidy-alphabetical-start
#![feature(rustc_private)]
// `mach` is built up incrementally and not yet wired into codegen; silence churn-y lints during
// development. These will be tightened once the module is consumed by the codegen path.
#![allow(dead_code)]
#![allow(unreachable_pub)]
// tidy-alphabetical-end

extern crate rustc_abi;
extern crate rustc_ast;
extern crate rustc_codegen_ssa;
extern crate rustc_data_structures;
extern crate rustc_errors;
extern crate rustc_hir;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_target;

// This links the backend against the same `rustc_driver` dylib as the host rustc process so that
// the rustc-private crates above resolve to the host's copies rather than being duplicated.
#[allow(unused_extern_crates)]
extern crate rustc_driver;

use std::any::Any;
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;

mod asm;
mod builder;
mod common;
mod constant;
mod consts;
mod context;
mod debuginfo;
mod declare;
mod mach;
mod type_;

use rustc_ast::expand::allocator::AllocatorMethod;
use rustc_codegen_ssa::back::lto::ThinModule;
use rustc_codegen_ssa::back::write::{
    CodegenContext, FatLtoInput, ModuleConfig, SharedEmitter, TargetMachineFactoryFn, ThinLtoInput,
};
use rustc_codegen_ssa::mono_item::MonoItemExt;
use rustc_codegen_ssa::traits::{CodegenBackend, ExtraBackendMethods, WriteBackendMethods};
use rustc_codegen_ssa::{CompiledModule, CompiledModules, CrateInfo, ModuleCodegen};
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_middle::dep_graph::{WorkProduct, WorkProductMap};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::config::{OptLevel, OutputFilenames, OutputType};
use rustc_span::Symbol;

use crate::builder::Builder;
use crate::context::CodegenCx;
use crate::mach::module::MachModule;

/// The unit of code handed from the (tcx-bearing) `compile_codegen_unit` front-half to the
/// (tcx-free) object-emitting `codegen` back-half on a worker thread. Must be `Send + Sync`.
#[derive(Default)]
pub struct Arm64Module {
    pub mach: MachModule,
}

/// The codegen backend. It implements `CodegenBackend` (the entry point rustc calls) as well as
/// `ExtraBackendMethods`/`WriteBackendMethods` (the per-codegen-unit driver used by
/// `rustc_codegen_ssa`'s generic MIR lowering).
#[derive(Clone)]
pub struct Arm64CodegenBackend;

impl CodegenBackend for Arm64CodegenBackend {
    fn name(&self) -> &'static str {
        "arm64"
    }

    fn thin_lto_supported(&self) -> bool {
        false
    }

    fn target_config(&self, sess: &Session) -> rustc_codegen_ssa::TargetConfig {
        // AArch64 mandates NEON; on Apple platforms AES/SHA2/SHA3 are also enabled by default.
        let mut target_features = vec![rustc_span::sym::neon];
        if sess.target.os == rustc_target::spec::Os::MacOs {
            target_features.extend([
                rustc_span::sym::aes,
                rustc_span::sym::sha2,
                rustc_span::sym::sha3,
            ]);
        }
        rustc_codegen_ssa::TargetConfig {
            target_features: target_features.clone(),
            unstable_target_features: target_features,
            has_reliable_f16: true,
            has_reliable_f16_math: true,
            has_reliable_f128: true,
            has_reliable_f128_math: true,
        }
    }

    fn target_cpu(&self, sess: &Session) -> String {
        match sess.opts.cg.target_cpu {
            Some(ref name) => name,
            None => sess.target.cpu.as_ref(),
        }
        .to_owned()
    }

    fn codegen_crate(&self, tcx: TyCtxt<'_>) -> Box<dyn Any> {
        Box::new(rustc_codegen_ssa::base::codegen_crate(self.clone(), tcx))
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        _outputs: &OutputFilenames,
        crate_info: &CrateInfo,
    ) -> (CompiledModules, WorkProductMap) {
        ongoing_codegen
            .downcast::<rustc_codegen_ssa::back::write::OngoingCodegen<Arm64CodegenBackend>>()
            .expect("expected Arm64CodegenBackend's OngoingCodegen, found Box<dyn Any>")
            .join(sess, crate_info)
    }
}

impl ExtraBackendMethods for Arm64CodegenBackend {
    type Module = Arm64Module;

    fn codegen_allocator(
        &self,
        _tcx: TyCtxt<'_>,
        _module_name: &str,
        _methods: &[AllocatorMethod],
    ) -> Self::Module {
        // TODO: emit the allocator shim (__rust_alloc and friends). Empty for now; programs that
        // actually allocate will fail to link until this is implemented.
        Arm64Module::default()
    }

    fn compile_codegen_unit(
        &self,
        tcx: TyCtxt<'_>,
        cgu_name: Symbol,
    ) -> (ModuleCodegen<Self::Module>, u64) {
        let cgu = tcx.codegen_unit(cgu_name);
        let mut cx = CodegenCx::new(tcx, cgu_name);
        let mono_items = cgu.items_in_deterministic_order(tcx);

        // Phase 1: predefine every item's symbol.
        for &(mono_item, data) in &mono_items {
            mono_item.predefine::<Builder<'_, '_>>(
                &mut cx,
                cgu_name.as_str(),
                data.linkage,
                data.visibility,
            );
        }

        // Phase 2: lower each item. A function populates `cur_fn`; finish it into the module.
        for &(mono_item, item_data) in &mono_items {
            if let rustc_middle::mono::MonoItem::Fn(instance) = mono_item {
                cx.cur_instance.set(Some(instance));
            }
            mono_item.define::<Builder<'_, '_>>(&mut cx, cgu_name.as_str(), item_data);
            if let Some(fb) = cx.cur_fn.borrow_mut().take() {
                cx.module.borrow_mut().push_function(fb.finish());
            }
            cx.cur_instance.set(None);
        }

        let mach = std::mem::take(&mut *cx.module.borrow_mut());
        let module = ModuleCodegen::new_regular(cgu_name.as_str().to_owned(), Arm64Module { mach });
        (module, 0)
    }
}

impl WriteBackendMethods for Arm64CodegenBackend {
    type Module = Arm64Module;
    type TargetMachine = ();
    type ModuleBuffer = Infallible;
    type ThinData = Infallible;

    fn target_machine_factory(
        &self,
        _sess: &Session,
        _opt_level: OptLevel,
        _target_features: &[String],
    ) -> TargetMachineFactoryFn<Self> {
        Arc::new(|_, _| ())
    }

    fn optimize_and_codegen_fat_lto(
        _sess: &Session,
        _cgcx: &CodegenContext,
        _shared_emitter: &SharedEmitter,
        _tm_factory: TargetMachineFactoryFn<Self>,
        _exported_symbols_for_lto: &[String],
        _each_linked_rlib_for_lto: &[PathBuf],
        _modules: Vec<FatLtoInput<Self>>,
    ) -> CompiledModule {
        unreachable!("LTO is not supported by rustc_codegen_arm64")
    }

    fn run_thin_lto(
        _cgcx: &CodegenContext,
        _prof: &SelfProfilerRef,
        _dcx: rustc_errors::DiagCtxtHandle<'_>,
        _exported_symbols_for_lto: &[String],
        _each_linked_rlib_for_lto: &[PathBuf],
        _modules: Vec<ThinLtoInput<Self>>,
    ) -> (Vec<ThinModule<Self>>, Vec<WorkProduct>) {
        unreachable!("LTO is not supported by rustc_codegen_arm64")
    }

    fn optimize(
        _cgcx: &CodegenContext,
        _prof: &SelfProfilerRef,
        _shared_emitter: &SharedEmitter,
        _module: &mut ModuleCodegen<Self::Module>,
        _config: &ModuleConfig,
    ) {
        // No optimizations: that is the entire point of this backend.
    }

    fn optimize_and_codegen_thin(
        _cgcx: &CodegenContext,
        _prof: &SelfProfilerRef,
        _shared_emitter: &SharedEmitter,
        _tm_factory: TargetMachineFactoryFn<Self>,
        _thin: ThinModule<Self>,
    ) -> CompiledModule {
        unreachable!("LTO is not supported by rustc_codegen_arm64")
    }

    fn codegen(
        cgcx: &CodegenContext,
        _prof: &SelfProfilerRef,
        _shared_emitter: &SharedEmitter,
        module: ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
    ) -> CompiledModule {
        let name = module.name.clone();
        let kind = module.kind;
        let mach = &module.module_llvm.mach;

        let object = cgcx.output_filenames.temp_path_for_cgu(OutputType::Object, &name);
        std::fs::write(&object, crate::mach::emit_obj::emit_object(mach))
            .expect("failed to write object file");

        let assembly = if config.emit_asm {
            let path = cgcx.output_filenames.temp_path_for_cgu(OutputType::Assembly, &name);
            std::fs::write(&path, crate::mach::emit_asm::emit_asm(mach))
                .expect("failed to write assembly file");
            Some(path)
        } else {
            None
        };

        CompiledModule {
            name,
            kind,
            object: Some(object),
            global_asm_object: None,
            dwarf_object: None,
            bytecode: None,
            assembly,
            llvm_ir: None,
            links_from_incr_cache: Vec::new(),
        }
    }

    fn serialize_module(_module: Self::Module, _is_thin: bool) -> Self::ModuleBuffer {
        unreachable!("LTO is not supported by rustc_codegen_arm64")
    }
}

/// Entry point: rustc loads this backend dynamically via `-Zcodegen-backend` and calls this symbol.
#[unsafe(no_mangle)]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    Box::new(Arm64CodegenBackend)
}
