//! Binary emission: turn a [`MachModule`] into a Mach-O object file via the `object` crate.
//!
//! Relocations use the explicit `RelocationFlags::MachO` form (the only correct choice for arm64);
//! see `RelocKind` for the mapping to `ARM64_RELOC_*`.

use std::collections::HashMap;

use object::write::{
    Object, Relocation, SectionId, StandardSection, Symbol, SymbolFlags, SymbolId, SymbolSection,
};
use object::{Architecture, BinaryFormat, Endianness, RelocationFlags, SectionFlags, SectionKind, SymbolKind, SymbolScope};

use crate::dwarf::{DebugContext, DebugRelocTarget, FnDebug};
use crate::mach::func::{Reloc, RelocKind};
use crate::mach::module::{DataSection, MachModule};

/// Emit `module` as a little-endian arm64 Mach-O object, returning the file bytes. When `debug` is
/// present, its DWARF sections are appended to the `__DWARF` segment with relocations against the
/// emitted function and DWARF-section symbols.
pub fn emit_object(module: &MachModule, debug: Option<DebugContext>) -> Vec<u8> {
    let mut obj = Object::new(BinaryFormat::MachO, Architecture::Aarch64, Endianness::Little);
    // Symbol names are stored already in final form (with the Mach-O leading underscore), so do no
    // additional mangling.
    obj.set_mangling(object::write::Mangling::None);

    // Emit an `LC_BUILD_VERSION` load command so the linker doesn't warn about a missing platform.
    // The minimum-OS version is the target's deployment target, captured from the session when the
    // module was built. The SDK version is omitted (`0`) \u2014 it does not influence the object and is
    // re-specified when linking the final binary, matching what rustc/LLVM do.
    let mut build_version = object::write::MachOBuildVersion::default();
    build_version.platform = object::macho::PLATFORM_MACOS;
    build_version.minos = module.macho_min_os;
    build_version.sdk = 0;
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
    // Byte offset of each function within `__text`, aligned with `encoded` (for DWARF addresses).
    let mut fn_text_offsets: Vec<u64> = Vec::with_capacity(encoded.len());
    for f in &encoded {
        let offset = obj.append_section_data(text, &f.code, 4);
        fn_text_offsets.push(offset);
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
        // Thread-local data needs the macOS TLV layout: the initializer in `__thread_data` plus a
        // 3-word descriptor in `__thread_vars` (`{__tlv_bootstrap, 0, &init}`). The descriptor is
        // built by hand rather than via `object`'s helper, which names the bootstrap symbol
        // `_tlv_bootstrap` (one underscore) whereas dyld provides `__tlv_bootstrap`.
        if d.section == DataSection::Tls {
            let tls_section = obj.section_id(StandardSection::Tls);
            let data_offset = obj.append_section_data(tls_section, &d.bytes, d.align as u64);
            let init_name = format!("{}$tlv$init", d.name);
            let init_id = obj.add_symbol(Symbol {
                name: init_name.into_bytes(),
                value: data_offset,
                size: d.bytes.len() as u64,
                kind: SymbolKind::Tls,
                scope: SymbolScope::Compilation,
                weak: false,
                section: SymbolSection::Section(tls_section),
                flags: SymbolFlags::None,
            });
            pending.push((tls_section, data_offset, &d.relocs));

            let vars_section = obj.section_id(StandardSection::TlsVariables);
            let desc_offset = obj.append_section_data(vars_section, &[0u8; 24], 8);
            let desc_id = obj.add_symbol(Symbol {
                name: d.name.as_bytes().to_vec(),
                value: desc_offset,
                size: 24,
                kind: SymbolKind::Tls,
                scope: scope_for(d.is_global),
                weak: false,
                section: SymbolSection::Section(vars_section),
                flags: SymbolFlags::None,
            });
            symbols.insert(d.name.clone(), desc_id);
            let bootstrap = *symbols
                .entry("__tlv_bootstrap".into())
                .or_insert_with(|| add_undefined(&mut obj, "__tlv_bootstrap"));
            let unsigned = macho_flags(RelocKind::Unsigned64);
            obj.add_relocation(
                vars_section,
                Relocation { offset: desc_offset, symbol: bootstrap, addend: 0, flags: unsigned },
            )
            .expect("tlv bootstrap relocation");
            obj.add_relocation(
                vars_section,
                Relocation {
                    offset: desc_offset + 16,
                    symbol: init_id,
                    addend: 0,
                    flags: unsigned,
                },
            )
            .expect("tlv init relocation");
            continue;
        }
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

    // Pass 3: unwind tables. Non-EH functions get a 32-byte `__compact_unwind` entry in fp-frame
    // mode so libunwind can walk through them. Functions with EH call sites need the personality to
    // run, which compact frame mode cannot express; they get a DWARF-mode compact entry pointing at
    // an `__eh_frame` FDE (with personality + LSDA) plus a `__gcc_except_tab` LSDA.
    const UNWIND_ARM64_MODE_FRAME: u32 = 0x0400_0000;
    const UNWIND_ARM64_MODE_DWARF: u32 = 0x0300_0000;
    let unsigned = macho_flags(RelocKind::Unsigned64);
    let sub64 = macho_flags(RelocKind::Subtractor64);

    // Build the shared `__eh_frame` (one CIE, one FDE per EH function) first so compact entries can
    // reference FDE offsets. macOS uses pointer-sized (8-byte) pc/lsda fields resolved by SUBTRACTOR
    // + UNSIGNED pairs; the personality is a 4-byte GOT-relative pointer in the CIE.
    //
    // BLOCKER (eh_frame gated off): the table bytes, CIE/FDE layout, section flags (0x6800000b),
    // and all SUB/UNSIGNED pointer pairs are byte-identical to LLVM and link fine. The one piece
    // that does not is the CIE personality, which macOS ld requires as a GOT-indirect pointer
    // (DW_EH_PE 0x9b, ARM64_RELOC_POINTER_TO_GOT) whose field holds the pcrel delta `-pers_field`.
    // object 0.39.1 (the latest release) cannot emit this:
    //   * write_relocation has no aarch64 GotRelative mapping, so POINTER_TO_GOT can only be set via
    //     explicit RelocationFlags::MachO.
    //   * macho_adjust_addend only applies pcrel_offset for i386/x86_64, so the GOT field stays 0 and
    //     ld resolves it absolute -> "address 0x1FC0 not in any section".
    //   * a non-zero addend is written as ARM64_RELOC_ADDEND with r_symbolnum=addend, which ld
    //     rejects in eh_frame -> "r_symbolnum out of range". A preset field is read back as that
    //     same implicit addend, so it also becomes ADDEND. Direct/abs personality is rejected too.
    // Newer object: 0.39.1 IS latest; it does not fix this. Fixes: patch the personality field to
    // -pers_field in the final bytes (addend 0, no ADDEND reloc), or upstream aarch64 POINTER_TO_GOT.
    let any_eh = false && encoded.iter().any(|f| !f.call_sites.is_empty());
    let mut eh_bytes: Vec<u8> = Vec::new();
    let mut fde_off: Vec<Option<u32>> = vec![None; encoded.len()];
    let mut pers_field = 0usize;
    if any_eh {
        // CIE: aug "zPLR", code_align=1, data_align=-8, ret_reg=30. P=pcrel|sdata4(0x10) to a local
        // stub that tail-calls the personality; L/R=pcrel|sdata4(0x10). Init CFI: def_cfa r31,0.
        let cie = b"\x00\x00\x00\x00\x01zPLR\x00\x01\x78\x1e\x07\x9b";
        eh_bytes.extend_from_slice(&0x18u32.to_le_bytes());
        eh_bytes.extend_from_slice(cie);
        pers_field = eh_bytes.len();
        eh_bytes.extend_from_slice(&0u32.to_le_bytes()); // personality (GOT pcrel; field patched post-write)
        eh_bytes.extend_from_slice(&[0x10, 0x10, 0x0c, 0x1f, 0x00]);
    }

    let cu = obj.add_section(b"__LD".to_vec(), b"__compact_unwind".to_vec(), SectionKind::Debug);
    let mut except_id = 0usize;
    let mut except_sec: Option<SectionId> = None;
    let mut lsda_syms: Vec<Option<SymbolId>> = vec![None; encoded.len()];
    for (i, f) in encoded.iter().enumerate() {
        if f.call_sites.is_empty() { continue; }
        let bytes = build_lsda(&f.call_sites);
        let etab = *except_sec.get_or_insert_with(|| obj.add_section(b"__TEXT".to_vec(), b"__gcc_except_tab".to_vec(), SectionKind::ReadOnlyData));
        let off = obj.append_section_data(etab, &bytes, 4);
        let name = format!("GCC_except_table{except_id}");
        except_id += 1;
        let sid = obj.add_symbol(Symbol {
            name: name.into_bytes(), value: off, size: bytes.len() as u64,
            kind: SymbolKind::Data, scope: SymbolScope::Compilation, weak: false,
            section: SymbolSection::Section(etab), flags: SymbolFlags::None,
        });
        lsda_syms[i] = Some(sid);
        // FDE: len(4), cie_ptr(4), pc_begin(8), range(8), auglen=8(1), lsda(8), CFI. pc/lsda are
        // 8-byte fields whose value is symbol-current (SUBTRACTOR ltmp + UNSIGNED target).
        fde_off[i] = Some(eh_bytes.len() as u32);
        let fde_start = eh_bytes.len();
        eh_bytes.extend_from_slice(&0u32.to_le_bytes()); // length placeholder
        let cie_ptr = (eh_bytes.len() - 0) as u32; // dist from here back to CIE start (0)
        eh_bytes.extend_from_slice(&cie_ptr.to_le_bytes());
        eh_bytes.extend_from_slice(&0u64.to_le_bytes()); // pc_begin (8B; func-anchor via reloc pair)
        eh_bytes.extend_from_slice(&(f.code.len() as u64).to_le_bytes()); // range (8B)
        eh_bytes.push(8); // aug len = 8 (one pointer)
        eh_bytes.extend_from_slice(&0u64.to_le_bytes()); // lsda (8B; table-anchor via reloc pair)
        // CFI: advance4; def_cfa_off16; advance(rest); def_cfa r29,16; off r30@-8; off r29@-16.
        eh_bytes.extend_from_slice(&[0x44, 0x0e, 0x10, 0x48, 0x0c, 0x1d, 0x10, 0x9e, 0x01, 0x9d, 0x02]);
        while (eh_bytes.len() - fde_start) % 4 != 0 { eh_bytes.push(0); }
        let len = (eh_bytes.len() - fde_start - 4) as u32;
        eh_bytes[fde_start..fde_start + 4].copy_from_slice(&len.to_le_bytes());
    }
    if any_eh { eh_bytes.extend_from_slice(&0u32.to_le_bytes()); } // terminator
    while any_eh && eh_bytes.len() % 8 != 0 { eh_bytes.push(0); }
    let pers_slot = eh_bytes.len();
    if any_eh { eh_bytes.extend_from_slice(&0u64.to_le_bytes()); } // personality pointer slot

    for (i, f) in encoded.iter().enumerate() {
        let func_id = symbols[&f.name];
        let eh = any_eh && !f.call_sites.is_empty();
        let mut hdr = Vec::with_capacity(32);
        hdr.extend_from_slice(&0u64.to_le_bytes()); // func (reloc)
        hdr.extend_from_slice(&(f.code.len() as u32).to_le_bytes());
        let enc = if eh { UNWIND_ARM64_MODE_DWARF | fde_off[i].unwrap() } else { UNWIND_ARM64_MODE_FRAME };
        hdr.extend_from_slice(&enc.to_le_bytes());
        hdr.extend_from_slice(&0u64.to_le_bytes()); // personality (reloc, optional)
        hdr.extend_from_slice(&0u64.to_le_bytes()); // lsda (reloc, optional)
        let base = obj.append_section_data(cu, &hdr, 8);
        obj.add_relocation(cu, Relocation { offset: base, symbol: func_id, addend: 0, flags: unsigned }).unwrap();
        // DWARF-mode compact entries carry pers/lsda in the FDE, so no compact pers/lsda relocs.
    }

    // Emit __eh_frame section and its relocations now that all FDE offsets are known.
    if any_eh {
        let ehs = obj.add_section(b"__TEXT".to_vec(), b"__eh_frame".to_vec(), SectionKind::ReadOnlyData);
        // ld only treats the section as eh_frame with these flags: S_COALESCED + NO_TOC +
        // STRIP_STATIC_SYMS + LIVE_SUPPORT (0x6800000b).
        obj.section_mut(ehs).flags = SectionFlags::MachO { flags: 0x6800_000b };
        let ehbase = obj.append_section_data(ehs, &eh_bytes, 8);
        let sub32 = macho_flags(RelocKind::Subtractor32);
        let u32r = macho_flags(RelocKind::Unsigned32);
        let got = macho_flags(RelocKind::PointerToGot32);
        let pers = *symbols.entry("_rust_eh_personality".into()).or_insert_with(|| add_undefined(&mut obj, "_rust_eh_personality"));
        let _ = (sub32, u32r, pers_slot);
        obj.add_relocation(ehs, Relocation { offset: ehbase + pers_field as u64, symbol: pers, addend: 0, flags: got }).unwrap();
        for (i, f) in encoded.iter().enumerate() {
            let Some(fo) = fde_off[i] else { continue };
            let func_id = symbols[&f.name];
            let pc = ehbase + fo as u64 + 8; // pc_begin field (8B)
            let pc_a = obj.add_symbol(Symbol { name: format!("Lpc{i}").into_bytes(), value: pc, size: 0, kind: SymbolKind::Data, scope: SymbolScope::Compilation, weak: false, section: SymbolSection::Section(ehs), flags: SymbolFlags::None });
            obj.add_relocation(ehs, Relocation { offset: pc, symbol: pc_a, addend: 0, flags: sub64 }).unwrap();
            obj.add_relocation(ehs, Relocation { offset: pc, symbol: func_id, addend: 0, flags: unsigned }).unwrap();
            let lp = ehbase + fo as u64 + 25; // lsda field (8B)
            let lp_a = obj.add_symbol(Symbol { name: format!("Llsda{i}").into_bytes(), value: lp, size: 0, kind: SymbolKind::Data, scope: SymbolScope::Compilation, weak: false, section: SymbolSection::Section(ehs), flags: SymbolFlags::None });
            obj.add_relocation(ehs, Relocation { offset: lp, symbol: lp_a, addend: 0, flags: sub64 }).unwrap();
            obj.add_relocation(ehs, Relocation { offset: lp, symbol: lsda_syms[i].unwrap(), addend: 0, flags: unsigned }).unwrap();
        }
    }

    // Emit DWARF debug info (line table + subprogram DIEs) into the `__DWARF` segment.
    if let Some(debug) = debug {
        emit_debug_info(&mut obj, debug, &encoded, &fn_text_offsets, text);
    }

    let mut bytes = obj.write().expect("failed to write Mach-O object");
    if any_eh {
        let cie = b"\x18\x00\x00\x00\x00\x00\x00\x00\x01zPLR";
        if let Some(p) = bytes.windows(cie.len()).position(|w| w == cie) {
            let f = p + pers_field;
            bytes[f..f + 4].copy_from_slice(&(-(pers_field as i32)).to_le_bytes());
        }
    }
    bytes
}

/// Build a GCC-style LSDA (`.gcc_except_table`) cleanup table from a function's call sites.
/// Header: lpstart=omit, ttype=omit, call-site encoding=uleb128. Offsets are function-relative.
fn build_lsda(call_sites: &[crate::mach::func::CallSite]) -> Vec<u8> {
    fn uleb(out: &mut Vec<u8>, mut v: u64) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v != 0 { out.push(b | 0x80); } else { out.push(b); break; }
        }
    }
    let mut cs = Vec::new();
    for c in call_sites {
        uleb(&mut cs, c.start as u64);
        uleb(&mut cs, c.len as u64);
        uleb(&mut cs, c.lp as u64);
        // action 0 = cleanup; otherwise a 1-based index into the action table below (we emit a
        // single catch-all action record at index 1).
        uleb(&mut cs, if c.action == 0 { 0 } else { 1 });
    }
    let has_catch = call_sites.iter().any(|c| c.action != 0);
    let mut out = Vec::new();
    out.push(0xff); // landing-pad base: omit (defaults to function start)
    if has_catch {
        // ttype encoding: udata4 absolute. One catch-all (null) typeinfo entry follows. Action
        // table: {ar_filter=1, ar_next=0}. ttbase = bytes from after this uleb to table end.
        out.push(0x03);
        let cs_block = 1 + uleb_len(cs.len() as u64) + cs.len();
        let action = 2usize; // sleb 1, sleb 0
        let ttype = 4usize; // one null 4-byte type
        out_uleb(&mut out, (cs_block + action + ttype) as u64);
        out.push(0x01);
        uleb(&mut out, cs.len() as u64);
        out.extend_from_slice(&cs);
        out.push(0x01); // ar_filter = 1
        out.push(0x00); // ar_next = 0
        out.extend_from_slice(&[0, 0, 0, 0]); // catch-all typeinfo (null)
    } else {
        out.push(0xff); // ttype encoding: omit (cleanup-only)
        out.push(0x01); // call-site encoding: uleb128, function-relative
        uleb(&mut out, cs.len() as u64);
        out.extend_from_slice(&cs);
    }
    out
}

fn uleb_len(mut v: u64) -> usize {
    let mut n = 1;
    while v >= 0x80 { v >>= 7; n += 1; }
    n
}
fn out_uleb(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 { out.push(b | 0x80); } else { out.push(b); break; }
    }
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
        DataSection::Tls => StandardSection::Tls,
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
        RelocKind::TlvpPage21 => (object::macho::ARM64_RELOC_TLVP_LOAD_PAGE21, true, 2),
        RelocKind::TlvpPageOff12 => (object::macho::ARM64_RELOC_TLVP_LOAD_PAGEOFF12, false, 2),
        RelocKind::Subtractor64 => (object::macho::ARM64_RELOC_SUBTRACTOR, false, 3),
        RelocKind::Subtractor32 => (object::macho::ARM64_RELOC_SUBTRACTOR, false, 2),
        RelocKind::Unsigned32 => (object::macho::ARM64_RELOC_UNSIGNED, false, 2),
        RelocKind::PointerToGot32 => (object::macho::ARM64_RELOC_POINTER_TO_GOT, true, 2),
    };
    RelocationFlags::MachO { r_type, r_pcrel, r_length }
}

/// Append the DWARF sections produced by `debug` to the `__DWARF` segment, translating gimli's
/// collected relocations into Mach-O `ARM64_RELOC_UNSIGNED` relocations against the `__text` and
/// DWARF-section symbols.
fn emit_debug_info(
    obj: &mut Object<'_>,
    mut debug: DebugContext,
    encoded: &[crate::mach::func::EncodedFunction],
    fn_text_offsets: &[u64],
    text: SectionId,
) {
    let fn_debugs: Vec<FnDebug<'_>> = encoded
        .iter()
        .zip(fn_text_offsets)
        .map(|(f, &off)| FnDebug {
            name: &f.name,
            text_offset: off,
            size: f.code.len() as u64,
            line_rows: &f.line_rows,
        })
        .collect();
    let sections = debug.emit(&fn_debugs);

    let text_sym = obj.section_symbol(text);

    // Create every DWARF section first (so cross-section references resolve), recording each gimli
    // section id -> (object section id, its section symbol).
    let mut section_map: HashMap<gimli::SectionId, (SectionId, SymbolId)> = HashMap::new();
    for sect in &sections {
        let name = sect.id.name().replace('.', "__"); // ".debug_info" -> "__debug_info"
        let kind = match sect.id {
            gimli::SectionId::DebugStr | gimli::SectionId::DebugLineStr => SectionKind::DebugString,
            _ => SectionKind::Debug,
        };
        let sid = obj.add_section(b"__DWARF".to_vec(), name.into_bytes(), kind);
        obj.section_mut(sid).set_data(sect.bytes.clone(), 1);
        let ssym = obj.section_symbol(sid);
        section_map.insert(sect.id, (sid, ssym));
    }

    // Apply relocations. `ARM64_RELOC_UNSIGNED` stores the addend inline in the field, so a
    // section-symbol target plus an offset addend is the correct encoding for both the code
    // references (low_pc / line-program addresses, against `__text`) and the inter-section
    // references (e.g. `__debug_info` -> `__debug_str`).
    for sect in &sections {
        let sid = section_map[&sect.id].0;
        for r in &sect.relocs {
            let (symbol, addend) = match r.target {
                DebugRelocTarget::Function(idx) => {
                    (text_sym, fn_text_offsets[idx] as i64 + r.addend)
                }
                DebugRelocTarget::Section(gid) => (section_map[&gid].1, r.addend),
            };
            obj.add_relocation(
                sid,
                Relocation {
                    offset: r.offset as u64,
                    symbol,
                    addend,
                    flags: macho_dwarf_flags(r.size),
                },
            )
            .expect("failed to add DWARF relocation");
        }
    }
}

/// Mach-O `ARM64_RELOC_UNSIGNED` flags for an absolute DWARF reference of `size` bytes.
fn macho_dwarf_flags(size: u8) -> RelocationFlags {
    RelocationFlags::MachO {
        r_type: object::macho::ARM64_RELOC_UNSIGNED,
        r_pcrel: false,
        r_length: size.trailing_zeros() as u8, // 8 -> 3, 4 -> 2, 2 -> 1, 1 -> 0
    }
}
