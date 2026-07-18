# How We Work — building `rustc_codegen_arm64` to production grade

This is the working process behind the baseline AArch64 backend. It exists so any session (human or
agent, on any host) can continue the work with the same rigor. The backend has found and fixed a
long list of **silent miscompiles** precisely because of this process — not despite it.

> Platform note: the commands below are written for the macOS dev host where the backend was built.
> On the arm64 Linux box, substitute the Linux equivalents called out in §11.

---

## 1. Principles

1. **Correctness is not self-evident in a codegen backend.** Wrong code compiles fine and runs wrong.
   We never trust "it looks right" — we prove it against an oracle (§4) or byte-compare (§7).
2. **Fail loud, never silent.** An unimplemented feature must `todo!`/`fatal` with a clear message,
   never emit plausible-but-wrong code. A loud gap is a backlog item; a silent miscompile is a bug
   that ships. Load-bearing fallbacks (e.g. the size fallbacks in `mem_size`/`fp_size` that the
   vector/ABI paths rely on) are the exception — do not "harden" them into panics.
3. **The oracle is LLVM.** For any program, `rustc` with the default LLVM backend defines correct
   behavior. Our job is to match it (or, for permitted non-determinism like signed-zero `min`/`max`,
   to knowingly differ — see §6).
4. **Refactors are byte-identical.** A change that claims to preserve behavior must produce
   byte-for-byte identical objects. If it doesn't, it's a behavior change and must be validated as one.
5. **Every fix ships with a test that would have caught it** (§8).
6. **Small, verifiable increments.** Change one thing, `./x check`, validate, then continue. Don't
   accumulate unrunnable code.
7. **Measure, don't guess** (for performance — §9) and **minimize churn** (touch only what the change
   requires; a tight diff keeps the regression anchor meaningful).

---

## 2. Backend design invariants (context that shapes the process)

- **`MachFunction` is a self-contained, position-independent atom.** It references everything external
  (other functions, statics) only by symbol via relocations, never by absolute offset. This is what
  makes future per-function parallel codegen and binary-patching incremental compilation possible.
  Don't introduce cross-function absolute assumptions.
- **Front-half / back-half split.** The front-half (`compile_codegen_unit`, has `TyCtxt`) lowers MIR
  into a `MachModule`. The back-half (`WriteBackendMethods::codegen`, a worker thread, **no `TyCtxt`**)
  serializes it to an object. Anything the emitter needs must be captured front-half and travel inside
  `MachModule` (this is why, e.g., `macho_min_os` is a field).
- **Two emitters kept in lockstep.** `mach/emit_obj.rs` (binary) and `mach/emit_asm.rs` (textual)
  must agree instruction-for-instruction; they are cross-checked by assembling the `.s` and diffing
  bytes against our own encoder (§4, tier 1).

---

## 3. The build / check / validate loop

| Step | Command | Notes |
|------|---------|-------|
| Type-check (fast) | `./x check cg_arm64` | ~1–4 s. First gate after every edit. |
| Build backend | `./x build cg_arm64 --stage 2` | dylib → `build/<host>/stage2-codegen/<host>/release/librustc_codegen_arm64.dylib` |
| **Restore std** | `./x build --stage 2 library` | **REQUIRED after building the backend** |
| Canonical validate | `./x test assembly-arm64 --stage 2 --force-rerun` | rebuilds + installs + runs FileCheck |

**The load-bearing gotcha:** `./x build cg_arm64` (or `./x test cg_arm64`) **recreates the stage
sysroot and WIPES `std` and the installed backend dylib.** Always follow a backend build with
`./x build --stage 2 library` to restore std, and re-install the dylib (the `assembly-arm64` test
step does the install for you; otherwise copy it into
`.../stage2/lib/rustlib/<host>/codegen-backends/`).

The dev toolchain `rg-arm64` points at the installed backend, so `rustc +rg-arm64
-Zcodegen-backend=arm64 …` uses it. You can also load a specific dylib by absolute path.

---

## 4. The validation pyramid (fastest → highest-signal)

1. **Encoder unit tests / fuzz vs `llvm-mc`** (`mach/emit_asm.rs` test module). Renders thousands of
   instructions, assembles with `llvm-mc --show-encoding`, compares to our `Inst::encode`. Catches
   raw encoding bugs. Extend by adding instruction families to `gen_insts()`.
2. **`assembly-arm64` FileCheck suite** (`tests/assembly-arm64/`). Golden-ish structural checks on
   emitted asm for specific features. **Every bug fix expressible as asm gets a case here.**
3. **Differential corpus + flag matrix** (`arm64-difftest/`, §5). Real programs compiled with LLVM
   *and* our backend across opt-level × overflow-checks, stdout/exit diffed. The workhorse for
   behavior.
4. **Real crate test suites.** Clone real crates (smallvec, arrayvec, vec-map, …), run their tests
   under both backends, compare. Highest-signal for real-world code.
5. **`abi-cafe` + cross-ABI programs.** ABI bugs are invisible to a single self-consistent program
   (both sides agree on the wrong convention). Catch them with LLVM↔arm64 cross-calls and abi-cafe.

Rule of thumb: **a single-program diff cannot catch an ABI bug.** Use tier 5 for calling-convention
work (critical for the Linux varargs delta).

---

## 5. The differential harness (`~/Code/arm64-difftest/`, sibling of the repo)

```
corpus/        + run-diff.sh        # 18 programs × opt{0,2,3} × overflow{on,off} vs LLVM (panic=abort)
unwind/        + run-unwind.sh      # panic=unwind: drop order, nested, catch_unwind, threads
locals/        + run-locals.sh      # debuginfo: lldb variable reads + `dwarfdump --verify`
inline-asm/                         # inline/global asm stress vs LLVM
crates/        + run-crates.sh      # real crate test suites under both backends
incremental/   + run-incremental.sh # incremental-compile correctness
symbench/      + gen.py + bench.sh  # perf benchmark generator + runner (§9)
```

Corpus discipline:
- **The corpus must be deterministic.** Remove anything with permitted non-determinism (e.g. `f32/f64`
  `min`/`max`/`clamp` on signed zeros — docs say either input may be returned). A flaky corpus is
  worse than none.
- Seed the corpus with a **minimized reproducer of every bug found** — it becomes a permanent
  regression test.
- Floats are compared via `to_bits()` for exact equality.

On the Linux box, this harness gets recreated/pointed at the box's LLVM `rustc` as the oracle and the
box's native linker.

---

## 6. Bug-hunting methodology

- **Differential probing:** generate broad corner-case batches (int all-widths, float↔int matrices,
  i128 torture, enums/niches, aggregates, ABI shapes), compile under both backends, diff. The
  highest-yield areas historically: **`f128` and `i128` special-cased paths** (least-tested types),
  **ABI**, and **anything with a "16-byte value" path** (a scalar fallback silently handling only the
  low 8 bytes is the classic silent-miscompile shape).
- **Minimal repro of the specific value.** When a diff fires, reproduce the *exact* diverging input in
  a 3–5 line program before theorizing.
- **Beware misleading diffs.** A divergence shifts subsequent line alignment, so the diff's *pairing*
  can point at the wrong line. Verify with a side-by-side (`paste`) and a minimal repro of the
  specific case — don't trust the diff's line pairing.
- **Audit siblings.** When you find a bug in one op (e.g. `fneg` using the scalar f64 path for f128),
  grep for the same pattern across all siblings (`fabs`, `copysign`, `select`, `store`, `ret`, …) —
  the bugs cluster.
- **Distinguish a bug from a gap.** If the toolchain (and LLVM itself) can't do it either (e.g. macOS
  `f128` libm), it's a documented gap, not our bug; omit from the corpus.

---

## 7. Refactor discipline (byte-identity)

For any change that should preserve behavior:
1. Save the pre-change backend dylib (e.g. `/tmp/cg_before.dylib`).
2. Build the post-change dylib (`/tmp/cg_after.dylib`).
3. Emit objects with **both** across a flag matrix (opt 0/2/3 × debuginfo 0/2 × overflow on/off) over
   the corpus + unwind + inline-asm + locals sources, and `cmp` each pair.
4. **All pairs must be byte-identical.** Any difference means it's a behavior change — validate it as
   one (§4) instead.

This is how the internal-representation refactors are proven safe (e.g. a symbol-handle change was
verified 124/124 byte-identical before being judged on its merits).

---

## 8. Every fix ships with a covering test

- Expressible as asm → add a `tests/assembly-arm64/<name>.rs` FileCheck case asserting the key
  instructions (e.g. `adcs`/`sbcs` present for i128 saturating).
- Behavioral → add a minimized reproducer to `arm64-difftest/corpus/` (or `unwind/`, etc.).
- The regression suite only earns trust if it grows with every bug.

---

## 9. Performance work methodology

- **Frontend dominates.** On typical code, borrowck + typeck + trait-solving + malloc churn dwarf the
  backend (~15–20% of non-idle work even on backend-heavy input). Backend micro-opts matter for
  backend-heavy workloads (e.g. compiling all of rustc), not for typical wall-time. Set expectations
  accordingly before optimizing.
- **Profile the backend in isolation:** use a *fast* (LLVM-built) `rustc` + the backend dylib, not the
  slow self-hosted one. On macOS, `/usr/bin/sample <pid> <dur> 1 -file out -mayDie` resolves and
  demangles Rust symbols with no install; filter to `rustc_codegen_arm64` frames for self-time.
  (Codegen runs on worker threads, so `-Ztime-passes` won't show it — use a sampler.)
- **A/B allocation counts** with a `malloc` interposer (`DYLD_INSERT_LIBRARIES` on macOS /
  `LD_PRELOAD` on Linux) for a clean, low-noise before/after on alloc traffic.
- **Benchmark signal:** wall time is noisy; **instructions retired** (`/usr/bin/time -l` on macOS,
  `perf stat` on Linux) is the low-noise proxy for work done. Report **min and median over N runs**
  (N≥25), and always confirm the **output object is byte-identical** so you're measuring the same work.
- **Judge honestly.** A measured win that isn't worth the complexity gets reverted (that has happened
  more than once). Record the measurement and the verdict either way.

---

## 10. Notes & memory discipline

- Keep a running progress log with the latest N entries summarized at the top; record the *lesson*
  from each bug (root cause + the general pattern), not just the fix.
- Record verified facts about the codebase (build commands, gotchas, ABI details, the macOS surface)
  as durable repo notes so the next session doesn't re-derive them.
- Update or delete notes that turn out wrong.

---

## 11. macOS → Linux tool equivalents (for the box)

| Purpose | macOS (dev host) | Linux (box) |
|---------|------------------|-------------|
| Object inspection | `otool`, `object` crate | `readelf -a`, `objdump -dr`, `nm` |
| Timing / instructions | `/usr/bin/time -l` | `/usr/bin/time -v`, `perf stat` |
| Sampling profiler | `/usr/bin/sample` | `perf record`/`report` |
| malloc A/B | `DYLD_INSERT_LIBRARIES` | `LD_PRELOAD` |
| Disassemble bytes | `llvm-mc`, `otool -tv` | `llvm-mc`, `objdump -d` |
| Link | `cc` (ld64) | `cc` (ld/lld/gold) |

The *method* is identical across platforms; only the tool names change. Whatever the tool, the loop is
always: **change → check → build+restore std → validate against the oracle (or byte-compare) → add a
covering test → record the lesson.**
