//! DWARF debug-info generation (Phase 1).
//!
//! Builds, per codegen unit, a single DWARF compile unit containing one `DW_TAG_subprogram` per
//! defined function plus a `.debug_line` line-number program, so lldb/dsymutil can recover function
//! names, file/line locations, and set breakpoints by line. There are no type or variable DIEs yet.
//!
//! The design mirrors `rustc_codegen_cranelift`'s debuginfo: gimli's `write` API produces the
//! section bytes and a list of relocations (collected by [`WriterRelocate`]); the object emitter
//! ([`crate::mach::emit_obj`]) turns those into Mach-O `__DWARF` sections.
//!
//! Because the object-emitting half of the backend runs without a `TyCtxt`, every span is resolved
//! to a concrete `(file, line, column)` here, in the `TyCtxt`-bearing half, and the finished
//! [`DebugContext`] travels with the module to be emitted later.

use gimli::write::{
    Address, AttributeValue, DwarfUnit, EndianVec, Expression, FileId, LineProgram, LineString,
    Range, RangeList, Result as GimliResult, Sections, UnitEntryId, Writer,
};
use gimli::{Encoding, Format, LineEncoding, Register, RunTimeEndian, SectionId};
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::ty::{Instance, TyCtxt};
use rustc_span::{Pos, SourceFile, Span, StableSourceFileId};

use crate::mach::inst::DebugLoc;

/// AArch64 DWARF register number for the frame pointer (`x29`). The backend's prologue establishes
/// `x29` as a stable frame base (`mov x29, sp` after saving the frame record), so it is used
/// directly as `DW_AT_frame_base` without needing call-frame information.
const DWARF_REG_X29: u16 = 29;

/// Per-function debug state recorded when its subprogram DIE is created. `low_pc`/`high_pc` are
/// filled in later (during [`DebugContext::emit`]) once the function's address and size are known.
struct FnEntry {
    die: UnitEntryId,
    decl: DebugLoc,
}

/// What a collected DWARF relocation points at. Resolved to a concrete Mach-O symbol by the object
/// emitter.
#[derive(Clone)]
pub enum DebugRelocTarget {
    /// The function at this index in the slice passed to [`DebugContext::emit`] (its address in
    /// `__text`).
    Function(usize),
    /// Another DWARF section (a section-relative reference, e.g. `__debug_info` -> `__debug_str`).
    Section(SectionId),
}

/// A relocation within a DWARF section, in gimli-neutral form.
#[derive(Clone)]
pub struct DebugReloc {
    pub offset: u32,
    pub size: u8,
    pub target: DebugRelocTarget,
    pub addend: i64,
}

/// A finished DWARF section: its gimli id, the raw bytes, and the relocations to apply.
pub struct DwarfSection {
    pub id: SectionId,
    pub bytes: Vec<u8>,
    pub relocs: Vec<DebugReloc>,
}

/// Information about one emitted function needed to finalize its debug info: the laid-out address
/// (as an offset within `__text`), the code size, and the resolved line markers.
pub struct FnDebug<'a> {
    pub name: &'a str,
    pub text_offset: u64,
    pub size: u64,
    pub line_rows: &'a [(u64, DebugLoc)],
}

/// All DWARF state for one codegen unit. Built in the `TyCtxt`-bearing half of the backend and then
/// carried (it is `Send`) to the object-emitting half.
pub struct DebugContext {
    endian: RunTimeEndian,
    dwarf: DwarfUnit,
    /// File table: index (as stored in [`DebugLoc::file`]) -> gimli line-program file id.
    file_ids: Vec<FileId>,
    /// Cache so each source file is registered once.
    file_cache: FxHashMap<StableSourceFileId, u32>,
    /// Subprogram DIEs keyed by the function's final (mangled) symbol name.
    functions: FxHashMap<Box<str>, FnEntry>,
}

impl DebugContext {
    /// Create a debug context for this codegen unit, or `None` if debug info is disabled.
    pub fn new(tcx: TyCtxt<'_>) -> Option<DebugContext> {
        use rustc_session::config::DebugInfo;
        if tcx.sess.opts.debuginfo == DebugInfo::None {
            return None;
        }

        // macOS debuggers default to DWARF <= 4.
        let encoding = Encoding { format: Format::Dwarf32, version: 4, address_size: 8 };

        let comp_dir = tcx
            .sess
            .source_map()
            .working_dir()
            .local_path()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        let producer = "rustc_codegen_arm64";
        let name = tcx.crate_name(rustc_hir::def_id::LOCAL_CRATE).to_string();

        let mut dwarf = DwarfUnit::new(encoding);

        let line_program = LineProgram::new(
            encoding,
            LineEncoding::default(),
            LineString::new(comp_dir.as_bytes(), encoding, &mut dwarf.line_strings),
            LineString::new(name.as_bytes(), encoding, &mut dwarf.line_strings),
            None,
        );
        dwarf.unit.line_program = line_program;

        {
            let producer_id = dwarf.strings.add(producer);
            let name_id = dwarf.strings.add(name);
            let comp_dir_id = dwarf.strings.add(comp_dir);
            let root = dwarf.unit.root();
            let root = dwarf.unit.get_mut(root);
            root.set(gimli::DW_AT_producer, AttributeValue::StringRef(producer_id));
            root.set(gimli::DW_AT_language, AttributeValue::Language(gimli::DW_LANG_Rust));
            root.set(gimli::DW_AT_name, AttributeValue::StringRef(name_id));
            // Placeholder; gimli fills the real line-program offset when writing.
            root.set(gimli::DW_AT_stmt_list, AttributeValue::Udata(0));
            root.set(gimli::DW_AT_comp_dir, AttributeValue::StringRef(comp_dir_id));
            root.set(gimli::DW_AT_low_pc, AttributeValue::Address(Address::Constant(0)));
        }

        Some(DebugContext {
            endian: RunTimeEndian::Little,
            dwarf,
            file_ids: Vec::new(),
            file_cache: FxHashMap::default(),
            functions: FxHashMap::default(),
        })
    }

    /// The compile-unit root DIE id (used as the scope for top-level functions and as a fallback).
    pub fn root(&self) -> UnitEntryId {
        self.dwarf.unit.root()
    }

    /// Register a source file (idempotent), returning its index in [`DebugContext::file_ids`].
    fn add_file(&mut self, source_file: &SourceFile) -> u32 {
        if let Some(&idx) = self.file_cache.get(&source_file.stable_id) {
            return idx;
        }
        let encoding = self.dwarf.unit.line_program.encoding();
        let path = source_file.name.prefer_local_unconditionally().to_string();
        let path = std::path::Path::new(&path);
        let (dir, file) = match (path.parent(), path.file_name()) {
            (Some(dir), Some(file)) if !dir.as_os_str().is_empty() => (
                Some(dir.to_string_lossy().into_owned()),
                file.to_string_lossy().into_owned(),
            ),
            (_, Some(file)) => (None, file.to_string_lossy().into_owned()),
            _ => (None, path.to_string_lossy().into_owned()),
        };

        let line_program = &mut self.dwarf.unit.line_program;
        let dir_id = match dir {
            Some(dir) => {
                let dir = LineString::new(dir.into_bytes(), encoding, &mut self.dwarf.line_strings);
                line_program.add_directory(dir)
            }
            None => line_program.default_directory(),
        };
        let file_name = LineString::new(file.into_bytes(), encoding, &mut self.dwarf.line_strings);
        let file_id = self.dwarf.unit.line_program.add_file(file_name, dir_id, None);

        let idx = self.file_ids.len() as u32;
        self.file_ids.push(file_id);
        self.file_cache.insert(source_file.stable_id, idx);
        idx
    }

    /// Resolve a span to a concrete `(file, line, column)` location.
    pub fn source_loc(&mut self, tcx: TyCtxt<'_>, span: Span) -> DebugLoc {
        let sm = tcx.sess.source_map();
        match sm.lookup_line(span.lo()) {
            Ok(rustc_span::SourceFileAndLine { sf, line }) => {
                let file = self.add_file(&sf);
                let line_pos = sf.lines()[line];
                let col = sf.relative_position(span.lo()) - line_pos;
                DebugLoc { file, line: line as u32 + 1, col: col.to_u32() + 1 }
            }
            Err(sf) => {
                let file = self.add_file(&sf);
                DebugLoc { file, line: 0, col: 0 }
            }
        }
    }

    /// Create a `DW_TAG_subprogram` DIE for a function defined in this codegen unit. `mangled_name`
    /// is the function's final symbol name (matching its `MachFunction`), used to attach the
    /// address range during emission.
    pub fn define_function<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        instance: Instance<'tcx>,
        mangled_name: &str,
    ) -> UnitEntryId {
        let def_id = instance.def_id();
        let decl = self.source_loc(tcx, tcx.def_span(def_id));

        // `def_path_str` uses the path-trimming machinery, which asserts it is only reached from a
        // diagnostic; suppress trimming since this is codegen, not a diagnostic.
        let display_name =
            rustc_middle::ty::print::with_no_trimmed_paths!(tcx.def_path_str(def_id));
        let linkage_name = tcx.symbol_name(instance).name;

        let name_id = self.dwarf.strings.add(display_name);
        let linkage_id = self.dwarf.strings.add(linkage_name);

        let root = self.dwarf.unit.root();
        let die = self.dwarf.unit.add(root, gimli::DW_TAG_subprogram);
        {
            let entry = self.dwarf.unit.get_mut(die);
            // Real values are filled in during `emit`, once the address/size are known.
            entry.set(gimli::DW_AT_low_pc, AttributeValue::Address(Address::Constant(0)));
            entry.set(gimli::DW_AT_high_pc, AttributeValue::Udata(0));

            let mut frame_base = Expression::new();
            frame_base.op_reg(Register(DWARF_REG_X29));
            entry.set(gimli::DW_AT_frame_base, AttributeValue::Exprloc(frame_base));

            entry.set(gimli::DW_AT_linkage_name, AttributeValue::StringRef(linkage_id));
            entry.set(gimli::DW_AT_name, AttributeValue::StringRef(name_id));
            entry.set(gimli::DW_AT_decl_file, AttributeValue::FileIndex(Some(self.file_ids[decl.file as usize])));
            entry.set(gimli::DW_AT_decl_line, AttributeValue::Udata(decl.line as u64));
            entry.set(gimli::DW_AT_external, AttributeValue::FlagPresent);
        }

        self.functions.insert(mangled_name.into(), FnEntry { die, decl });
        die
    }

    /// Finalize the line-number program and address ranges, then serialize all DWARF sections.
    pub fn emit(&mut self, funcs: &[FnDebug<'_>]) -> Vec<DwarfSection> {
        let mut ranges = Vec::new();

        for (i, f) in funcs.iter().enumerate() {
            let Some(entry) = self.functions.get(f.name) else { continue };
            let (die, decl) = (entry.die, entry.decl);
            let low_pc = Address::Symbol { symbol: i, addend: 0 };

            {
                let entry = self.dwarf.unit.get_mut(die);
                entry.set(gimli::DW_AT_low_pc, AttributeValue::Address(low_pc));
                entry.set(gimli::DW_AT_high_pc, AttributeValue::Udata(f.size));
            }
            ranges.push(Range::StartLength { begin: low_pc, length: f.size });

            let file_ids = &self.file_ids;
            let lp = &mut self.dwarf.unit.line_program;
            lp.begin_sequence(Some(low_pc));
            // Ensure the prologue (before the first marker) maps to the function's declaration line.
            if f.line_rows.first().map(|&(off, _)| off != 0).unwrap_or(true) {
                push_row(lp, file_ids, 0, decl);
            }
            let mut last_off = None;
            for &(off, loc) in f.line_rows {
                if last_off == Some(off) {
                    // Multiple markers at the same address: keep the last by overwriting the row.
                    set_row(lp, file_ids, off, loc);
                    continue;
                }
                push_row(lp, file_ids, off, loc);
                last_off = Some(off);
            }
            lp.end_sequence(f.size);
        }

        let range_list = self.dwarf.unit.ranges.add(RangeList(ranges));
        let root = self.dwarf.unit.root();
        self.dwarf.unit.get_mut(root).set(gimli::DW_AT_ranges, AttributeValue::RangeListRef(range_list));

        let mut sections = Sections::new(WriterRelocate::new(self.endian));
        self.dwarf.write(&mut sections).expect("gimli DWARF write failed");

        let mut out = Vec::new();
        sections
            .for_each_mut(|id, section| -> GimliResult<()> {
                let bytes = section.writer.slice().to_vec();
                if !bytes.is_empty() {
                    out.push(DwarfSection {
                        id,
                        bytes,
                        relocs: std::mem::take(&mut section.relocs),
                    });
                }
                Ok(())
            })
            .expect("gimli section iteration failed");
        out
    }
}

/// Set the current line-program row's fields without emitting it.
fn set_row(lp: &mut LineProgram, file_ids: &[FileId], offset: u64, loc: DebugLoc) {
    let row = lp.row();
    row.address_offset = offset;
    row.file = file_ids[loc.file as usize];
    row.line = loc.line as u64;
    row.column = loc.col as u64;
}

/// Set and emit a line-program row.
fn push_row(lp: &mut LineProgram, file_ids: &[FileId], offset: u64, loc: DebugLoc) {
    set_row(lp, file_ids, offset, loc);
    lp.generate_row();
}

/// A gimli [`Writer`] that writes into a byte buffer while collecting the relocations needed for
/// symbol and section references (which gimli writes as zeroes).
#[derive(Clone)]
struct WriterRelocate {
    relocs: Vec<DebugReloc>,
    writer: EndianVec<RunTimeEndian>,
}

impl WriterRelocate {
    fn new(endian: RunTimeEndian) -> WriterRelocate {
        WriterRelocate { relocs: Vec::new(), writer: EndianVec::new(endian) }
    }
}

impl Writer for WriterRelocate {
    type Endian = RunTimeEndian;

    fn endian(&self) -> RunTimeEndian {
        self.writer.endian()
    }

    fn len(&self) -> usize {
        self.writer.len()
    }

    fn write(&mut self, bytes: &[u8]) -> GimliResult<()> {
        self.writer.write(bytes)
    }

    fn write_at(&mut self, offset: usize, bytes: &[u8]) -> GimliResult<()> {
        self.writer.write_at(offset, bytes)
    }

    fn write_address(&mut self, address: Address, size: u8) -> GimliResult<()> {
        match address {
            Address::Constant(val) => self.write_udata(val, size),
            Address::Symbol { symbol, addend } => {
                let offset = self.len() as u32;
                self.relocs.push(DebugReloc {
                    offset,
                    size,
                    target: DebugRelocTarget::Function(symbol),
                    addend,
                });
                self.write_udata(0, size)
            }
        }
    }

    fn write_offset(&mut self, val: usize, section: SectionId, size: u8) -> GimliResult<()> {
        let offset = self.len() as u32;
        self.relocs.push(DebugReloc {
            offset,
            size,
            target: DebugRelocTarget::Section(section),
            addend: val as i64,
        });
        self.write_udata(0, size)
    }

    fn write_offset_at(
        &mut self,
        offset: usize,
        val: usize,
        section: SectionId,
        size: u8,
    ) -> GimliResult<()> {
        self.relocs.push(DebugReloc {
            offset: offset as u32,
            size,
            target: DebugRelocTarget::Section(section),
            addend: val as i64,
        });
        self.write_udata_at(offset, 0, size)
    }
}
