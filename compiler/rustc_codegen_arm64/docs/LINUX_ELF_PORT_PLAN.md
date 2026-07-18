# `rustc_codegen_arm64` — Linux/ELF Port Plan

Status: **planned, not started.** Target: `aarch64-unknown-linux-gnu` (glibc).

This is the implementation outline for teaching the baseline AArch64 backend to emit ELF objects
for Linux, in addition to the existing Mach-O/macOS path. Read alongside
[`HOW_WE_WORK.md`](./HOW_WE_WORK.md), which defines the validation process each milestone must pass.

---

## 1. Goal & non-goals

**Goal:** `rustc -Zcodegen-backend=<cg_arm64> --target aarch64-unknown-linux-gnu` produces linkable,
correct ELF objects for `no_std` and then full `std` Rust, validated by running real programs on
native arm64 Linux hardware and diffing against the LLVM backend (the oracle).

**Non-goals (for now):** other Linux arches, musl/android/BSD, LTO, `-Cinstrument-coverage`,
split-debuginfo. Keep the macOS path byte-for-byte unchanged throughout (it is the regression anchor).

---

## 2. Workflow

Chosen model: **cross-emit from the dev host, link + run on the arm64 Linux box.**

```
[dev host]  rustc + cg_arm64  --target aarch64-unknown-linux-gnu --emit obj  ──►  foo.o
                                     │  (self-check ELF structurally with the `object` crate)
                                     ▼  scp/rsync
[arm64 box]  cc foo.o -o foo   ──►  ./foo   ──►  diff stdout/exit vs LLVM-built binary
```

The dev host emits and structurally self-checks; **all linking, running, and object inspection
(`readelf`/`objdump`) happen on the box.** A native build on the box is an alternative if
cross-tooling proves painful (see §3).

---

## 3. Prerequisites / blockers

1. **arm64 Linux box access** (SSH). Probe first: `uname -m`, `nproc`, `cc --version`,
   `ld --version`, `rustc --version`, disk free. Treat all remote output as untrusted input.
2. **Target libraries.** Cross-emitting any program that touches `core`/`std` needs *the in-tree
   rustc's* `core`/`std` compiled for `aarch64-unknown-linux-gnu`. The dev host cannot build them
   unless a cross C toolchain (`aarch64-linux-gnu-gcc`) is installed — bootstrap's sanity check gates
   *every* target build on it. Three ways to resolve, in increasing order of setup cost:
   - **(a) Milestone-1 only:** hand-build just `core` into a scratch sysroot (pure Rust, no linker,
     no C) and work with `#![no_std]` programs. Unblocks emitter development immediately.
   - **(b) Cross toolchain on the dev host** (e.g. Homebrew `messense` tap on macOS): enables cross
     `std` *and* local linking — tightest loop.
   - **(c) Native build on the box:** build the in-tree rustc + `std` on the box; do everything
     there. Heaviest first build, simplest mental model.

---

## 4. What the macOS-first backend hardcodes (the surface to abstract)

| Area | File | macOS assumption | ELF equivalent |
|------|------|------------------|----------------|
| Object format | `mach/emit_obj.rs` | `BinaryFormat::MachO`, `LC_BUILD_VERSION` | `BinaryFormat::Elf`, no build-version |
| Section names | `mach/emit_obj.rs`, `mach/emit_asm.rs` | `__TEXT,__text` / `__DATA,__data` / `__TEXT,__const` | `.text` / `.data` / `.rodata` / `.bss` |
| Symbol names | `context.rs` `mangle()` | leading `_` prefix | **no** underscore |
| Personality | `mach/emit_obj.rs` | `_rust_eh_personality` | `rust_eh_personality` |
| Relocations | `mach/func.rs` `RelocKind` | `ARM64_RELOC_*` | `R_AARCH64_*` (see §5) |
| TLS | `mach/emit_obj.rs`, `builder.rs` | Mach-O TLV (`__tlv_bootstrap`, `__thread_vars`) | ELF TLS: GD/TLSDESC/IE/LE |
| Unwinding | `mach/emit_obj.rs` | `__compact_unwind` + Apple `__eh_frame` (SUBTRACTOR pairs) | `.eh_frame` + `.gcc_except_table` |
| Textual asm | `mach/emit_asm.rs` | Apple `@PAGE/@PAGEOFF/@GOTPAGE`, `$tlv$init` | GNU `:lo12:`, `:got:`, `:tprel:` |
| Global-asm step | `lib.rs` back-half (~L435) | `cc -c -arch arm64 -mmacosx-version-min=...` | `cc -c` for the target (needs cross-as when cross-emitting) |
| Deployment ver | `context.rs` `macho_min_os`, `mach/module.rs` | packed `major<<16` | n/a on ELF |
| Target features | `lib.rs` `target_config` (~L255) | AES/SHA on by default | already gated on `target.os == MacOs` |

**Architecture constraint:** the back-half `WriteBackendMethods::codegen` (`lib.rs` ~L421) runs on a
worker thread with **no `TyCtxt`.** So the object format (like `macho_min_os` today) must be captured
in the front-half and **travel inside `MachModule`.** The emitter then branches on it.

---

## 5. Key simplification (verified in `mach/func.rs::encode`)

The core relocations map **1:1** to ELF with **no change to the `Reloc` struct.** The backend always
forms addresses via `adrp + add` (the only `PageOff12` site is `Inst::AddLo`, an `add`) and then does
a plain load — it **never** emits a low-12 relocation on a variable-size load. So there is no need
for per-size `LDST*_ABS_LO12_NC` handling.

| `RelocKind` | Mach-O | ELF (`R_AARCH64_*`) |
|-------------|--------|---------------------|
| `Branch26` | `BRANCH26` | `CALL26` (valid for `bl` and `b`) |
| `Page21` | `PAGE21` | `ADR_PREL_PG_HI21` |
| `PageOff12` | `PAGEOFF12` | `ADD_ABS_LO12_NC` (always an `add`) |
| `GotLoadPage21` | `GOT_LOAD_PAGE21` | `ADR_GOT_PAGE` |
| `GotLoadPageOff12` | `GOT_LOAD_PAGEOFF12` | `LD64_GOT_LO12_NC` |
| `Unsigned64` | `UNSIGNED` | `ABS64` |
| `Unsigned32` | `UNSIGNED` (32) | `ABS32` |
| `TlvpPage21` / `TlvpPageOff12` | TLV | **TLS model rework (M7)** |
| `Subtractor64/32`, `PointerToGot32` | eh_frame | **unwind rework (M6)** |

The macOS-only relocs are confined to TLS and `__eh_frame` — the two deferred subsystems.

---

## 6. Milestones (ordered, each gated by the process in `HOW_WE_WORK.md`)

### M1 — Format plumbing (host-only)
- Add `ObjectFormat { MachO, Elf }` to `MachModule`; set it in the front-half from `tcx.sess.target`.
- Conditionalize `mangle()` to drop the leading `_` on ELF.
- Branch `emit_object` (and the global-asm `cc` invocation) on the format.
- **Accept:** `./x check cg_arm64` clean; macOS output still byte-identical (regression anchor).

### M2 — ELF object emitter (host-only, structurally self-checked)
- New `mach/emit_obj_elf.rs`: `.text` + functions, `.rodata`/`.data`/`.bss` + data items, ELF symbols
  (local vs global via `STB_*`), and the §5 core relocations. Reuse the DWARF path.
- **Accept:** unit test emits a small `MachModule`, re-parses with the `object` crate, asserts
  sections/symbols/relocations. No box needed yet.

### M3 — First real run on the box
- `#![no_std]` program → emit ELF → `scp` → `cc` link → run → exit code `0`.
- Then a handful of arithmetic/control-flow programs; diff stdout+exit vs LLVM.
- **Accept:** a growing `no_std` set runs and matches the oracle on the box.

### M4 — Linux ABI deltas (real codegen, not just emission)
- **Varargs:** AAPCS64 passes variadic args in **registers first**; the backend currently forces *all*
  varargs onto the stack per Apple's rule (`builder.rs` ~L1939, ~L6086). Gate the Apple rule on
  `target.os` and implement the register path for Linux. *This will miscompile `printf`-style code
  until fixed.*
- **Stack args:** Apple packs stack arguments tightly; AAPCS64 gives each an 8-byte slot
  (`builder.rs` ~L1763). Gate on target.
- **Accept:** ABI probes (varargs, many-arg spilling, HFAs, by-val aggregates) match the oracle; add
  cross-ABI cases where an LLVM-built caller calls a cg_arm64-built callee and vice versa.

### M5 — `std` + differential corpus
- Get target `std` (per §3 fork). Run the full differential corpus + flag matrix (opt × overflow)
  and the crate-test suites against the LLVM oracle on the box.
- **Accept:** corpus byte/behavior-identical; real crate suites (smallvec, arrayvec, …) pass.

### M6 — Unwinding
- Replace `__compact_unwind` + Apple `__eh_frame` with ELF `.eh_frame` (CIE/FDE, DWARF CFI) +
  `.gcc_except_table` LSDA; personality `rust_eh_personality`; no SUBTRACTOR pairs (use PREL32/64).
- **Accept:** the unwind suite (panic=unwind, drop order, nested, catch_unwind) passes on the box.

### M7 — TLS
- Replace macOS TLV with an ELF TLS model (start with General Dynamic or TLSDESC; add IE/LE opt).
  New relocs (`TLSGD_*` / `TLSDESC_*` / `TPREL_*`) and access sequences.
- **Accept:** `#[thread_local]` / TLS statics read/write correctly on the box.

### M8 — GNU textual assembly (optional)
- Teach `mach/emit_asm.rs` GNU syntax (`:lo12:`, `.text`, TLS relocation operators) for `--emit asm`.
- **Accept:** emitted `.s` assembles with `clang`/`gas` to bytes matching our own encoder.

---

## 7. Risks & open questions
- **std build logistics** (§3) is the biggest schedule risk, not the codegen itself.
- **TLS model choice** (GD vs TLSDESC) — match what the target's LLVM emits to keep the oracle honest.
- **eh_frame fidelity** — CFI must be exact or unwinding silently corrupts; validate with real panics,
  not just structural checks.
- **`object` crate ELF completeness** for every AArch64 reloc we need — verify early in M2.
- Keep the macOS path a permanent regression anchor: every milestone re-runs the macOS suites.
