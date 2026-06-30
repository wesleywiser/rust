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

/// A source location plus its lexical scope and inlined-at chain — the backing data for a
/// [`DebugLoc`] (`DILocation`). `scope` is the subprogram DIE the location belongs to (the abstract
/// instance for inlined code); `inlined_at`, when set, indexes the location table for the call site
/// in the enclosing frame. Walking the `inlined_at` chain reconstructs the inline-frame stack.
#[derive(Clone)]
struct LocData {
    scope: UnitEntryId,
    file: u32,
    line: u32,
    col: u32,
    inlined_at: Option<u32>,
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
    /// Location table indexed by [`DebugLoc::loc`]; records scope + inlined-at chain.
    locations: Vec<LocData>,
    /// Abstract subprogram DIEs for inlined callees, keyed by the callee's symbol name (one per
    /// callee per codegen unit, shared by every inlined instance / `DW_TAG_inlined_subroutine`).
    abstract_fns: FxHashMap<Box<str>, UnitEntryId>,
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
            locations: Vec::new(),
            abstract_fns: FxHashMap::default(),
        })
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

    /// Resolve a span to a concrete `(file, line, column)`, registering its source file.
    fn resolve_span(&mut self, tcx: TyCtxt<'_>, span: Span) -> (u32, u32, u32) {
        let sm = tcx.sess.source_map();
        match sm.lookup_line(span.lo()) {
            Ok(rustc_span::SourceFileAndLine { sf, line }) => {
                let file = self.add_file(&sf);
                let line_pos = sf.lines()[line];
                let col = sf.relative_position(span.lo()) - line_pos;
                (file, line as u32 + 1, col.to_u32() + 1)
            }
            Err(sf) => (self.add_file(&sf), 0, 0),
        }
    }

    /// Intern a location, returning the [`DebugLoc`] that references it.
    fn intern_location(&mut self, data: LocData) -> DebugLoc {
        let loc = self.locations.len() as u32;
        let dl = DebugLoc { file: data.file, line: data.line, col: data.col, loc };
        self.locations.push(data);
        dl
    }

    /// Resolve `span` within `scope` (optionally inlined at `inlined_at`) into a [`DebugLoc`].
    pub fn make_location(
        &mut self,
        tcx: TyCtxt<'_>,
        scope: UnitEntryId,
        inlined_at: Option<DebugLoc>,
        span: Span,
    ) -> DebugLoc {
        let (file, line, col) = self.resolve_span(tcx, span);
        self.intern_location(LocData { scope, file, line, col, inlined_at: inlined_at.map(|l| l.loc) })
    }

    /// Clone a location into a fresh, distinct [`DebugLoc`]. Used to give two inlinings that share a
    /// call-site span (e.g. from a macro) distinct identities so they form separate inline frames.
    pub fn clone_location(&mut self, loc: DebugLoc) -> DebugLoc {
        let data = self.locations[loc.loc as usize].clone();
        self.intern_location(data)
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
        let (decl_file, decl_line, decl_col) = self.resolve_span(tcx, tcx.def_span(def_id));

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
            entry.set(gimli::DW_AT_decl_file, AttributeValue::FileIndex(Some(self.file_ids[decl_file as usize])));
            entry.set(gimli::DW_AT_decl_line, AttributeValue::Udata(decl_line as u64));
            entry.set(gimli::DW_AT_external, AttributeValue::FlagPresent);
        }

        let decl = self.intern_location(LocData {
            scope: die,
            file: decl_file,
            line: decl_line,
            col: decl_col,
            inlined_at: None,
        });
        self.functions.insert(mangled_name.into(), FnEntry { die, decl });
        die
    }

    /// Create (or reuse) the abstract `DW_TAG_subprogram` DIE for a callee that is inlined into one
    /// or more functions in this codegen unit. It carries `DW_AT_inline` and no address range;
    /// each `DW_TAG_inlined_subroutine` references it via `DW_AT_abstract_origin`.
    pub fn define_abstract_function<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        callee: Instance<'tcx>,
    ) -> UnitEntryId {
        let key = tcx.symbol_name(callee).name;
        if let Some(&die) = self.abstract_fns.get(key) {
            return die;
        }
        let def_id = callee.def_id();
        let (decl_file, decl_line, _) = self.resolve_span(tcx, tcx.def_span(def_id));
        let display_name =
            rustc_middle::ty::print::with_no_trimmed_paths!(tcx.def_path_str(def_id));
        let name_id = self.dwarf.strings.add(display_name);
        let linkage_id = self.dwarf.strings.add(key);

        let root = self.dwarf.unit.root();
        let die = self.dwarf.unit.add(root, gimli::DW_TAG_subprogram);
        {
            let entry = self.dwarf.unit.get_mut(die);
            entry.set(gimli::DW_AT_name, AttributeValue::StringRef(name_id));
            entry.set(gimli::DW_AT_linkage_name, AttributeValue::StringRef(linkage_id));
            entry.set(gimli::DW_AT_decl_file, AttributeValue::FileIndex(Some(self.file_ids[decl_file as usize])));
            entry.set(gimli::DW_AT_decl_line, AttributeValue::Udata(decl_line as u64));
            entry.set(gimli::DW_AT_inline, AttributeValue::Inline(gimli::DW_INL_inlined));
        }
        self.abstract_fns.insert(key.into(), die);
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

            {
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

            // Reconstruct DW_TAG_inlined_subroutine frames from the per-instruction inline chain.
            self.build_inline_dies(die, i, f.line_rows, f.size);
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

    /// Reconstruct `DW_TAG_inlined_subroutine` DIEs for `func_die` (the concrete function at
    /// `fn_index`) from its line markers. Each marker carries an inlined-at chain (via the location
    /// table); instructions sharing an inline instance become one inlined-subroutine DIE — nested
    /// for multi-level inlining — carrying the inlined body's PC range(s), an `abstract_origin`
    /// reference to the callee's abstract subprogram, and the call-site file/line/column.
    fn build_inline_dies(
        &mut self,
        func_die: UnitEntryId,
        fn_index: usize,
        line_rows: &[(u64, DebugLoc)],
        fn_size: u64,
    ) {
        // Collapse runs of same-offset markers (keep the last, matching the line table) into
        // [start, end) segments tagged with the effective location.
        let mut segments: Vec<(u64, u64, u32)> = Vec::new();
        let mut k = 0;
        while k < line_rows.len() {
            let off = line_rows[k].0;
            let mut j = k;
            while j + 1 < line_rows.len() && line_rows[j + 1].0 == off {
                j += 1;
            }
            let end = if j + 1 < line_rows.len() { line_rows[j + 1].0 } else { fn_size };
            if end > off {
                segments.push((off, end, line_rows[j].1.loc));
            }
            k = j + 1;
        }

        let mut nodes: Vec<InlineNode> = Vec::new();
        let mut node_map: FxHashMap<(Option<usize>, UnitEntryId, u32), usize> = FxHashMap::default();

        for &(start, end, loc) in &segments {
            let inner = inline_instance(
                &mut self.dwarf,
                &self.locations,
                &self.file_ids,
                func_die,
                &mut nodes,
                &mut node_map,
                loc,
            );
            // The PC range belongs to the innermost inline instance and every ancestor (whose
            // inlined body contains it).
            let mut cur = inner;
            while let Some(idx) = cur {
                nodes[idx].ranges.push((start, end));
                cur = nodes[idx].parent;
            }
        }

        // Attach each inline instance's coalesced ranges as low/high_pc (single range) or
        // DW_AT_ranges (multiple).
        for node in &nodes {
            let merged = merge_ranges(&node.ranges);
            if merged.len() == 1 {
                let (s, e) = merged[0];
                let entry = self.dwarf.unit.get_mut(node.die);
                entry.set(
                    gimli::DW_AT_low_pc,
                    AttributeValue::Address(Address::Symbol { symbol: fn_index, addend: s as i64 }),
                );
                entry.set(gimli::DW_AT_high_pc, AttributeValue::Udata(e - s));
            } else {
                let list = merged
                    .iter()
                    .map(|&(s, e)| Range::StartLength {
                        begin: Address::Symbol { symbol: fn_index, addend: s as i64 },
                        length: e - s,
                    })
                    .collect();
                let rid = self.dwarf.unit.ranges.add(RangeList(list));
                self.dwarf
                    .unit
                    .get_mut(node.die)
                    .set(gimli::DW_AT_ranges, AttributeValue::RangeListRef(rid));
            }
        }
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

/// One inline instance under construction: its `DW_TAG_inlined_subroutine` DIE, its parent instance
/// (or `None` for an instance inlined directly into the function), and the PC ranges it covers.
struct InlineNode {
    die: UnitEntryId,
    parent: Option<usize>,
    ranges: Vec<(u64, u64)>,
}

/// Resolve the innermost inline instance for location `loc_idx`, lazily creating the
/// `DW_TAG_inlined_subroutine` DIE (and its ancestors). Returns `None` when the location is not
/// inlined (it belongs to the enclosing function directly). Instances are keyed by
/// `(parent, abstract callee, call-site location)` so all instructions of one inlining share a DIE.
fn inline_instance(
    dwarf: &mut DwarfUnit,
    locations: &[LocData],
    file_ids: &[FileId],
    func_die: UnitEntryId,
    nodes: &mut Vec<InlineNode>,
    node_map: &mut FxHashMap<(Option<usize>, UnitEntryId, u32), usize>,
    loc_idx: u32,
) -> Option<usize> {
    let call_idx = locations[loc_idx as usize].inlined_at?;
    // Lexical blocks collapse to their subprogram, so the location's scope is its (abstract) callee.
    let abstract_func = locations[loc_idx as usize].scope;
    let parent = inline_instance(dwarf, locations, file_ids, func_die, nodes, node_map, call_idx);

    let key = (parent, abstract_func, call_idx);
    if let Some(&n) = node_map.get(&key) {
        return Some(n);
    }
    let parent_die = match parent {
        Some(p) => nodes[p].die,
        None => func_die,
    };
    let die = dwarf.unit.add(parent_die, gimli::DW_TAG_inlined_subroutine);
    {
        let call = &locations[call_idx as usize];
        let entry = dwarf.unit.get_mut(die);
        entry.set(gimli::DW_AT_abstract_origin, AttributeValue::UnitRef(abstract_func));
        entry.set(gimli::DW_AT_call_file, AttributeValue::FileIndex(Some(file_ids[call.file as usize])));
        entry.set(gimli::DW_AT_call_line, AttributeValue::Udata(call.line as u64));
        entry.set(gimli::DW_AT_call_column, AttributeValue::Udata(call.col as u64));
    }
    let idx = nodes.len();
    nodes.push(InlineNode { die, parent, ranges: Vec::new() });
    node_map.insert(key, idx);
    Some(idx)
}

/// Sort and coalesce adjacent/overlapping `[start, end)` ranges.
fn merge_ranges(ranges: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut sorted = ranges.to_vec();
    sorted.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::new();
    for (s, e) in sorted {
        if let Some(last) = out.last_mut()
            && s <= last.1
        {
            last.1 = last.1.max(e);
            continue;
        }
        out.push((s, e));
    }
    out
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
