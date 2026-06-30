//! A whole codegen-unit's worth of machine code: the functions and data items that the emitters
//! turn into a Mach-O object (or textual assembly).
//!
//! Symbol names are stored exactly as they must appear in the final symbol table (including any
//! platform prefix such as the Mach-O leading underscore); the emitters do no name mangling.

use rustc_data_structures::fx::FxHashSet;

use crate::mach::func::{MachFunction, Reloc};

/// Where a data item lives, which selects its Mach-O section.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DataSection {
    /// Mutable data: `__DATA,__data`.
    Data,
    /// Read-only data: `__TEXT,__const`.
    ReadOnly,
    /// Zero-initialized data: `__DATA,__bss`.
    Bss,
    /// Thread-local data: `__DATA,__thread_data`. The object emitter also generates the
    /// `__thread_vars` descriptor (the `__tlv_bootstrap` entry) for the item's symbol.
    Tls,
}

/// A statically-allocated data item (a `static`, a constant aggregate, a string literal, ...).
pub struct DataItem {
    pub name: Box<str>,
    pub is_global: bool,
    pub section: DataSection,
    pub align: u32,
    /// Initializer bytes. For [`DataSection::Bss`] this is empty and `bss_size` gives the length.
    pub bytes: Vec<u8>,
    pub bss_size: u64,
    /// Pointers embedded in the data (e.g. a `&T` field), as symbol relocations.
    pub relocs: Vec<Reloc>,
}

/// All code and data produced for one codegen unit.
pub struct MachModule {
    pub functions: Vec<MachFunction>,
    pub data: Vec<DataItem>,
    /// Packed macOS deployment version (`major << 16 | minor << 8 | patch`) for the object's
    /// `LC_BUILD_VERSION` load command, captured from the session target. Defaults to 11.0, the
    /// first Apple-Silicon macOS, when no session is available.
    pub macho_min_os: u32,
    /// Symbol names that must be emitted with global (external) scope even if their item has
    /// internal linkage, because they are referenced by a `sym` operand from this codegen unit's
    /// `global_asm!`/`asm!` (which is assembled into a separate object that can only see globals).
    pub forced_globals: FxHashSet<Box<str>>,
}

impl MachModule {
    pub fn new() -> MachModule {
        MachModule {
            functions: Vec::new(),
            data: Vec::new(),
            macho_min_os: 0x000B_0000,
            forced_globals: FxHashSet::default(),
        }
    }

    pub fn push_function(&mut self, func: MachFunction) {
        self.functions.push(func);
    }

    pub fn push_data(&mut self, data: DataItem) {
        self.data.push(data);
    }
}

impl Default for MachModule {
    fn default() -> Self {
        MachModule::new()
    }
}
