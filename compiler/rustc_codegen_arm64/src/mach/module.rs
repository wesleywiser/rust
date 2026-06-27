//! A whole codegen-unit's worth of machine code: the functions and data items that the emitters
//! turn into a Mach-O object (or textual assembly).
//!
//! Symbol names are stored exactly as they must appear in the final symbol table (including any
//! platform prefix such as the Mach-O leading underscore); the emitters do no name mangling.

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
#[derive(Default)]
pub struct MachModule {
    pub functions: Vec<MachFunction>,
    pub data: Vec<DataItem>,
}

impl MachModule {
    pub fn new() -> MachModule {
        MachModule::default()
    }

    pub fn push_function(&mut self, func: MachFunction) {
        self.functions.push(func);
    }

    pub fn push_data(&mut self, data: DataItem) {
        self.data.push(data);
    }
}
