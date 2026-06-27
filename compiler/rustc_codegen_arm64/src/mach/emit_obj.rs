//! Binary emission: turn a [`MachModule`] into a Mach-O object file via the `object` crate.
//!
//! Relocations use the explicit `RelocationFlags::MachO` form (the only correct choice for arm64);
//! see `RelocKind` for the mapping to `ARM64_RELOC_*`.

use std::collections::HashMap;

use object::write::{
    Object, Relocation, SectionId, StandardSection, Symbol, SymbolFlags, SymbolId, SymbolSection,
};
use object::{Architecture, BinaryFormat, Endianness, RelocationFlags, SymbolKind, SymbolScope};

use crate::mach::func::{Reloc, RelocKind};
use crate::mach::module::{DataSection, MachModule};

/// Emit `module` as a little-endian arm64 Mach-O object, returning the file bytes.
pub fn emit_object(module: &MachModule) -> Vec<u8> {
    let mut obj = Object::new(BinaryFormat::MachO, Architecture::Aarch64, Endianness::Little);
    // Symbol names are stored already in final form (with the Mach-O leading underscore), so do no
    // additional mangling.
    obj.set_mangling(object::write::Mangling::None);

    // Emit an `LC_BUILD_VERSION` load command so the linker doesn't warn about a missing platform.
    // FIXME: derive the minimum OS / SDK version from the target instead of hard-coding 11.0.
    let mut build_version = object::write::MachOBuildVersion::default();
    build_version.platform = object::macho::PLATFORM_MACOS;
    build_version.minos = 0x000B_0000; // 11.0.0
    build_version.sdk = 0x000B_0000; // 11.0.0
    obj.set_macho_build_version(build_version);

    let text = obj.section_id(StandardSection::Text);

    // Maps a symbol name to its id, covering every symbol defined in this object. Used to resolve
    // relocation targets; names not found here are added as undefined (external) symbols on demand.
    let mut symbols: HashMap<Box<str>, SymbolId> = HashMap::new();
    // Records `(section, section_offset, relocs)` for each defined item so relocations can be added
    // after all symbols exist.
    let mut pending: Vec<(SectionId, u64, &[Reloc])> = Vec::new();

    // Pass 1: define all functions and data items (and their symbols). Functions are encoded up
    // front so the layout (and relocations) are available here.
    let encoded: Vec<_> = module.functions.iter().map(|f| f.encode()).collect();
    for f in &encoded {
        let offset = obj.append_section_data(text, &f.code, 4);
        let id = obj.add_symbol(Symbol {
            name: f.name.as_bytes().to_vec(),
            value: offset,
            size: f.code.len() as u64,
            kind: SymbolKind::Text,
            scope: scope_for(f.is_global),
            weak: false,
            section: SymbolSection::Section(text),
            flags: SymbolFlags::None,
        });
        symbols.insert(f.name.clone(), id);
        pending.push((text, offset, &f.relocs));
    }

    for d in &module.data {
        let section = obj.section_id(standard_section(d.section));
        let offset = if d.section == DataSection::Bss {
            obj.append_section_bss(section, d.bss_size, d.align as u64)
        } else {
            obj.append_section_data(section, &d.bytes, d.align as u64)
        };
        let size = if d.section == DataSection::Bss { d.bss_size } else { d.bytes.len() as u64 };
        let id = obj.add_symbol(Symbol {
            name: d.name.as_bytes().to_vec(),
            value: offset,
            size,
            kind: SymbolKind::Data,
            scope: scope_for(d.is_global),
            weak: false,
            section: SymbolSection::Section(section),
            flags: SymbolFlags::None,
        });
        symbols.insert(d.name.clone(), id);
        pending.push((section, offset, &d.relocs));
    }

    // Pass 2: add relocations, creating undefined symbols for any unresolved targets.
    for (section, base, relocs) in pending {
        for r in relocs {
            let target = *symbols
                .entry(r.sym.clone())
                .or_insert_with(|| add_undefined(&mut obj, &r.sym));
            obj.add_relocation(
                section,
                Relocation {
                    offset: base + r.offset,
                    symbol: target,
                    addend: r.addend,
                    flags: macho_flags(r.kind),
                },
            )
            .expect("failed to add relocation");
        }
    }

    obj.write().expect("failed to write Mach-O object")
}

fn add_undefined(obj: &mut Object<'_>, name: &str) -> SymbolId {
    obj.add_symbol(Symbol {
        name: name.as_bytes().to_vec(),
        value: 0,
        size: 0,
        kind: SymbolKind::Unknown,
        scope: SymbolScope::Dynamic,
        weak: false,
        section: SymbolSection::Undefined,
        flags: SymbolFlags::None,
    })
}

fn scope_for(is_global: bool) -> SymbolScope {
    if is_global { SymbolScope::Dynamic } else { SymbolScope::Compilation }
}

fn standard_section(section: DataSection) -> StandardSection {
    match section {
        DataSection::Data => StandardSection::Data,
        DataSection::ReadOnly => StandardSection::ReadOnlyData,
        DataSection::Bss => StandardSection::UninitializedData,
    }
}

fn macho_flags(kind: RelocKind) -> RelocationFlags {
    let (r_type, r_pcrel, r_length) = match kind {
        RelocKind::Branch26 => (object::macho::ARM64_RELOC_BRANCH26, true, 2),
        RelocKind::Page21 => (object::macho::ARM64_RELOC_PAGE21, true, 2),
        RelocKind::PageOff12 => (object::macho::ARM64_RELOC_PAGEOFF12, false, 2),
        RelocKind::GotLoadPage21 => (object::macho::ARM64_RELOC_GOT_LOAD_PAGE21, true, 2),
        RelocKind::GotLoadPageOff12 => (object::macho::ARM64_RELOC_GOT_LOAD_PAGEOFF12, false, 2),
        RelocKind::Unsigned64 => (object::macho::ARM64_RELOC_UNSIGNED, false, 3),
    };
    RelocationFlags::MachO { r_type, r_pcrel, r_length }
}
