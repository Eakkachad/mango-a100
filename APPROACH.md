# Approach & Engineering Guidelines (v2 — 2026-09-07)

v1 is archived at [`APPROACH.v1.md`](./APPROACH.v1.md). It proposed **Candle** as the serving
substrate. That was written before we knew the target model's architecture. Measurement moved
us: MANGO is a hybrid Gated-DeltaNet model with a built-in MTP head, and **llama.cpp is the
only runtime that supports `qwen35` with MTP/DFlash speculative decoding today**. Candle would
have meant re-implementing the model *and* the drafter before any experiment could run.

## Strategy in one paragraph

**Rust orchestrates; llama.cpp is the kernel library** (decision D1). llama.cpp runs the model
through its C API; katgpt-rs supplies the research components as ordinary Rust dependencies:
`caddtree_budget` (adaptive draft budget), `weaver` (draft-logit corrector), and
`gdn_tree_verify` (rollback-free tree verification against GDN recurrent layers). Per-step FFI
traffic is small — `h_verifier[4096]`, `h_dflash[D][4096]`, `dflash_logits[D][K]` — negligible
beside a forward pass. This keeps a real Rust/katgpt-rs contribution alive, per the owner's
stated priority, without re-implementing a serving stack.

## Component priority

1. **Baseline (done, Phase 1):** llama.cpp + the official GGUF, measured per language and per
   draft budget. This is the control every later claim is measured against.
2. **P2.3 — `caddtree_budget` driving the draft budget per request.** The cleanest contribution
   available: no training, no new model, and `results/P13c_FINDINGS.md` already quantifies the
   target (English optimum n=1, Thai optimum n=0 — a static config cannot serve both).
3. **P2.4 — `gdn_tree_verify` against the real model's 24 GDN layers.** Correctness first
   (does it agree with sequential verification?), then speed. E02 only proved it correct on
   synthetic trees against its own reference.
4. **P2.2 — Weaver on the real draft stream.** Blocked: no trained checkpoint at hidden=4096,
   and the training side (`riir-train`) is not in this workspace. Run it as the documented
   zero-weight no-op and say so, or obtain the checkpoint.
5. **Post-hackathon:** HLA distillation.

## Guidelines

- **Correctness before speed.** Never benchmark a path whose output has not been checked
  against a reference — exact match for logic/dequant, a perplexity-delta bound for lossy steps.
- **No FP8 on any A100-bound path.** A100 (Ampere) has no FP8 tensor cores. An Ada dev card
  (RTX 4060) does — that is a trap, not a feature. Keep FP8 behind a default-off flag if it
  exists at all.
- **Build llama.cpp from source.** Packaged b9960 has the nextn C API but not the
  `--spec-type` selector; every Phase-1 result came from the vendored b10068 tree.
  `scripts/build_llama.sh` handles it.
- **One harness, every host.** The same scripts run on M4, the RTX box and A100:
  `experiments/common_env.py` resolves binaries (`LLAMA_BIN_DIR`), `NGL` tunes offload, and
  `experiments/p14_energy/power.py` swaps `nvml` / `powermetrics` / `none` behind one interface.
  The *harness* is portable even though the *numbers* are not.
- **Tuning constants are per-host.** `n_max` is not a project constant — it tracks the host's
  FLOP/byte ratio. Re-sweep on every new machine; record it as a new findings file rather than
  editing an old one.
- **Report per language.** Thai costs 1.845× the tokens for the same content, so a single
  tok/s or tokens-per-joule figure is language-mix dependent. Publish both languages plus a
  content-normalised figure.
- **A100 queue discipline.** Every queued session runs a pre-written, non-interactive script.
  Take one cheap early session (does it load, does NVML read, does the script finish) rather
  than one big burn at the end.
- **Keep the docs current.** Update `PROGRESS.md` every session; update `AGENT_HANDOFF.md` §3/§7
  when a key fact or next-step changes. A stale handoff is worse than none.

## Definition of done (hackathon submission)

- MANGO1.5-Qwen3.5-9B served on A100, measured on real hardware (never extrapolated):
  tokens/sec, TTFT, inter-token latency, mean and peak GPU power, tokens-per-joule **with an
  error band**, and perplexity delta against an unquantized baseline — **reported per language**.
- The adaptive-budget result (P2.3) as a delta against the fixed-budget baseline on the same
  hardware.
- Stretch: `gdn_tree_verify` on the real model; Weaver with a trained checkpoint.
- Honest negative results are shipped, not hidden — the workspace standard set by
  `dg-sdlm/BENCHMARK_REPORT.md`'s correction notice. P1.3b→P1.3c is already an instance: the
  first conclusion ("speculation is a net loss") was an artefact of a default, and the
  correction is recorded rather than quietly overwritten.
