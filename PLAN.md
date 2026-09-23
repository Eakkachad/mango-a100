# Plan v2 — rewritten 2026-09-07 after E01–E04

v1 is kept at [`PLAN.v1.md`](./PLAN.v1.md) for provenance. This version replaces it because
four measured findings invalidated its assumptions:

| v1 assumed | E01–E04 measured | consequence |
|---|---|---|
| Target is a standard softmax-attention Qwen | **24/32 layers are Gated DeltaNet**, 8 are full attention | `gdn_tree_verify` is on-target, not speculative |
| We must train an EAGLE/Medusa draft head (5h–2d on A100) | Model **ships an MTP head**; llama.cpp already implements **DFlash drafting on it** | drafter is free; delete the training budget |
| AWQ quantization is our safety-net deliverable | **CMKL already publishes `AWQ-W4A16-mtp`** | AWQ is the starting line — remove as a deliverable |
| KV-cache quant is the low-risk win | Only **8/32 layers hold a KV cache** | payoff ~¼ of assumed; demote to stretch |

## Architecture decision (D1) — Rust orchestrates, llama.cpp is the kernel library

**Chosen: a Rust binary that links llama.cpp through its C API, and calls katgpt-rs crates
natively.** llama.cpp runs the model (it is the only runtime that supports `qwen35` +
DFlash/MTP today); katgpt-rs supplies Weaver, CaDDTree, and `gdn_tree_verify` as ordinary
Rust dependencies. Per-step FFI traffic is tiny (`h_verifier[4096]`, `h_dflash[D][4096]`,
`topk_ids[D][K]`, `dflash_logits[D][K]`) — negligible beside a forward pass.

Rejected alternatives: (a) rewriting Weaver/CaDDTree/gdn_tree_verify in C++ inside
llama.cpp — throws away the Rust research asset and diverges from katgpt-rs upstream;
(b) a Rust cdylib called *from* llama.cpp — inverts control and complicates the build for
no gain.

Precedent: `neural-engines/chimera` already vendors llama.cpp under a Rust project.

**Risk to retire first (P1.2):** the MTP entry points (`llama_set_embeddings_nextn`,
`llama_get_embeddings_nextn_ith`) live in `src/llama-ext.h`, described in-source as a
*staging API*. If they are not exported from the built library, D1 needs a patched build
of the vendored b10068 rather than the brew install. **Do not write integration code until
P1.2 answers this.**

---

## Phase 1 — Get a novel number without any integration ✅ COMPLETE (2026-09-07)

The point of Phase 1 is that it produces a publishable, hackathon-usable result **even if
the whole integration effort later fails**. It runs on the M4, on llama.cpp alone.

- **P1.1 — Does the installed llama.cpp expose DFlash/MTP speculation at runtime?**
  Inspect `--spec-draft-type` options and whether a draft context is required.
  Exit: a yes/no recorded in `results/`, and if no, a built binary from the vendored
  b10068 source that does.
- **P1.2 — Is the nextn API reachable from outside the library?** (retires the D1 risk)
  Exit: written verdict; if not exported, note exactly what patch is needed.
- **P1.3 — Thai vs. English acceptance-rate asymmetry.** *This is the novelty bet.*
  Hypothesis: Thai (no word delimiters, 248K-vocab tokenizer) yields systematically
  different MTP/DFlash draft acceptance than English on the same model. Nothing in the
  speculative-decoding literature reports bilingual acceptance behaviour — every EAGLE /
  Medusa / SpecDec / Cascade number surveyed is English Llama-family
  (see `knowledge-base/topics/speculative-decoding-and-pruning.md`).
  Build a matched Thai/English prompt set (same tasks, same lengths, translated pairs),
  measure per-language acceptance rate and tokens/step at several draft depths.
  Exit: a table + plot, and a stated effect size with its uncertainty.
- **P1.4 — Energy harness v0** (needed for every later claim; nothing like it exists in
  the workspace). Sample power at 100 ms with idle-baseline subtraction — the methodology
  both energy papers in the knowledge base converge on
  (`knowledge-base/topics/energy-measurement.md`). On the M4 use `powermetrics`; keep the
  NVML/`nvidia-smi --query-gpu=power.draw` path behind the same interface so the A100 run
  needs no rewrite. Report tokens-per-joule with an error band — nvidia-smi's real error
  is ±5% proportional, not the ±5W NVIDIA claims.

**Phase 1 exit criteria — all met.** Results: `results/P13a_FINDINGS.md` (token density
1.845×), `P13b_FINDINGS.md` (acceptance 70.5% en / 20.7% th), `P13c_FINDINGS.md` (budget
sweep: +19.5% at n_max=1 in English, no winning Thai budget), `P14_FINDINGS.md` (energy
harness, math-verified), `P1_FINDINGS.md` (nextn API exported but C++-mangled).

## Phase 1.5 — Re-establish on CUDA (NEW, inserted 2026-09-07)

Phase 1 ran entirely on the M4, whose numbers do not transfer. Before Phase 2 integration is
worth writing, the same ground must be re-taken on the GPU box — and the integration itself
should be built on CUDA, not Metal, or it gets written twice.

- **P1.5.1** Hardware is confirmed: **RTX 4060 8 GB, Windows 11** (Ada `sm_89`, 272 GB/s).
  Verify actual free VRAM under load — the Q4_K_M target fits in ~6.9 GB of a ~7 GB usable
  budget, so it is tight; the DFlash drafter does not fit alongside it (use `draft-mtp`), and
  the 9.1 GB AWQ build does not fit at all. **Never tune an FP8 path: Ada has FP8, A100 does not.**
- **P1.5.2** Build llama.cpp with CUDA (`scripts/build_llama.sh`), verify `--spec-type` lists
  `draft-mtp`.
- **P1.5.3** **Exercise the `nvml` energy backend for the first time** — it is written but has
  never touched real hardware, and it is the instrument every energy claim depends on.
- **P1.5.4** Re-run the budget sweep. **Expect the optimal `n_max` to be larger than the M4's
  1** — the RTX card's FLOP/byte is 7–12× the M4's. Record it as a new findings file rather
  than editing the M4 one.
- **P1.5.5** **≥3 repeats for error bars.** Do this here, not on the M4: these are the numbers
  that will actually be quoted.
- **P1.5.6** Widen the matrix: `draft-dflash` (drafter already identified), `ngram-*`, and a
  larger prompt set than the current 10 pairs.

**Exit criteria:** a CUDA budget-sweep table with error bars, a working NVML energy reading,
and a two-point (M4 + RTX) picture of how the optimal budget moves with FLOP/byte — enough to
predict the A100 optimum before spending queue time.

## Phase 2 — Integration: Weaver + CaDDTree on the real draft stream

- **P2.1** Rust bin that loads MANGO via llama.cpp C API and reproduces P1.3's numbers
  through the Rust path (equivalence check — same acceptance rate, or explain the delta).
- **P2.2** Feed the real `dflash_logits` / `h_dflash` into `WeaverCorrector`. Blocked on a
  Weaver checkpoint at hidden=4096 — **the training side (`riir-train`) is not in this
  workspace**; either obtain it, or run Weaver as the documented zero-weight no-op and
  report honestly that the corrector was untrained.
- **P2.3** Put `caddtree_budget` in charge of draft depth/width and measure acceptance and
  tokens/step against the fixed-budget baseline from P1.3. **This is the cleanest
  contribution that needs no new training**: same model, same drafter, adaptive budget.
- **P2.4** Wire `gdn_tree_verify` to verify draft trees against MANGO's 24 GDN layers —
  first correctness (does it agree with sequential verification on the real model?), then
  speed. Note E02 only proved it correct on synthetic trees against its own reference.

## Phase 3 — A100 validation (queued, scarce — scripts written in advance)

- **P3.1** Re-run the Phase 1 harness unchanged on A100 with `CMKL/…-AWQ-W4A16-mtp`,
  producing the real throughput / TTFT / ITL / watts / tokens-per-joule numbers.
- **P3.2** Re-run whichever Phase 2 stages landed, and report **deltas against the same
  baseline on the same hardware** — never stack-vs-vLLM absolutes.
- **P3.3** One early cheap sanity session before the big run (confirm the direction of
  each effect survives the hardware change) rather than one burn at the end.

## Scope commitments

**Committed (must ship):** P1.1–P1.4, P2.3, P3.1–P3.2. This yields: a bilingual
acceptance-rate finding, an energy methodology, and an adaptive-budget result measured on
real A100 hardware — none of which requires training anything.

**Stretch:** P2.2 (needs a Weaver checkpoint), P2.4 (GDN verify on the real model),
KV-cache quantization (demoted — only 8/32 layers cache), HLA distillation (unchanged:
post-hackathon research track).

**Explicitly dropped:** AWQ/SmoothQuant as a deliverable (CMKL ships it); EAGLE/Medusa
draft-head training (MTP replaces it).

## Claim discipline

Every number carries its hardware, and M4 numbers are never presented as A100 numbers.
Report deltas against an identical baseline, not absolutes against vLLM. Techniques
implemented from others' papers (`gdn_tree_verify` ← Oda et al. arXiv:2607.06763,
CaDDTree ← arXiv:2606.01813, BASTION ← arXiv:2605.29727) are cited as such — the
contribution claimed is the composition, the bilingual measurement, and the energy
characterisation. Weaver's originality is **unverified**: it mirrors the user's own
`riir-train` Plan 314 and cites arXiv:2607.06763 only for the surrounding framework, so a
prior-art search is required before claiming it as novel.
