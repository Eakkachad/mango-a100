# Plan

Anchor date: 2026-09-02 (planning complete). Hackathon starts ~2026-10-01. Dates below are targets, not
commitments — update them in `PROGRESS.md` as reality diverges, don't silently drift this file.

## Week 1 (Sep 2–8) — Foundation + deep-dive

**Goal:** a correct, benchmarked Candle+CUDA baseline; a real assessment of `katgpt-speculative`'s
portability.

- Deep-dive `katgpt-speculative` (Weaver module, `dd_tree.rs`, `caddtree_budget.rs`, `verifier_trait.rs`).
  Read the two cited papers. Decide: retarget Weaver from Gemma2-2B to Qwen3.5-9B — feasible this month,
  or a stretch item?
- Stand up MANGO1.5-Qwen3.5-9B (or a Qwen3.5-architecture proxy) on Candle + CUDA, INT4, on the 4060.
- Build the benchmark harness (tokens/sec, TTFT, ITL, perplexity delta; NVML power hook stubbed in, even
  if only exercised on A100 later).
- Confirm Qwen3.5's exact attention/norm/RoPE variant against Candle's existing model implementations —
  flag any gap that needs a custom implementation.

**Exit criteria:** baseline generates correct, coherent output on the 4060; harness runs end-to-end and
produces all logged metrics; go/no-go decision on Weaver retargeting is written down (not left implicit).

## Week 2 (Sep 9–15) — KV-cache quantization

**Goal:** port `katgpt-quant`'s codecs to CUDA kernels, validated on the 4060.

- Pick one codec first (start with whichever the Week 1 review flags as simplest — likely TurboQuant or
  PlanarQuant) and get it correct before attempting the rest.
- Measure compression ratio vs. perplexity delta on the baseline model.
- First queued A100 session: correctness + rough-direction sanity check only (not full benchmarking yet)
  — confirm the kernel still behaves as expected at A100 scale before investing further tuning time.

**Exit criteria:** at least one KV-quant codec running correctly on both GPUs; a documented
compression-ratio-vs-quality curve.

## Week 3 (Sep 16–22) — Speculative decoding

**Goal:** working draft-and-verify loop using the retargeted Weaver/DDTree system (if Week 1 said go), or
a from-scratch EAGLE/Medusa-style implementation as fallback (if Week 1 said the retarget isn't feasible
this month).

- Check the open question first: does a pretrained EAGLE/Medusa-style draft head already exist for a
  Qwen3.5-class model? If yes, skip training and go straight to serving integration.
- If training a draft head is needed: budget ~5 hours (Medusa-1-style) to ~1–2 days (EAGLE-style) on
  A100 — this is a case where using a queued A100 slot for actual training is justified, not just
  inference benchmarking.
- Measure token-acceptance rate on the 4060 (transfers directly, see `AGENT_HANDOFF.md` §5).

**Exit criteria:** measured acceptance rate and effective speedup direction confirmed; decision on whether
speculative decoding makes the final submission or stays a stretch item.

## Week 4 (Sep 23–29) — Integration, A100 validation, safety net

**Goal:** full pipeline assembled; real A100 numbers; AWQ/SmoothQuant safety net in place regardless of
how Weeks 2–3 landed.

- Integrate AWQ or SmoothQuant weight quantization (INT4/INT8) as the baseline quantization layer,
  independent of whatever katgpt-rs components made it in — this is the fallback that guarantees a
  defensible submission even if the novel components hit a wall.
- Dedicated A100 session(s) for the real numbers: throughput, TTFT, ITL, GPU power draw (NVML/CodeCarbon),
  tokens-per-joule, perplexity delta — across whatever combination of techniques is actually working.
- Batch-size / occupancy tuning pass specific to A100 (sm_80 launch configs).
- Freeze submission scope; write up results honestly (including what *didn't* work — see the "definition
  of done" note in `APPROACH.md` about honest negative results).

**Exit criteria:** submission-ready pipeline with real, on-hardware A100 numbers for every metric the
hackathon scores.

## Hackathon week (~Sep 30–Oct) — polish, pitch, live verification

- Final polish, pitch materials, any last live-A100 re-verification the queue allows.

## Post-hackathon — continuing research (not deadline-bound)

- HLA distillation: attempt to recover quality after swapping HLA in for MANGO1.5-Qwen3.5-9B's trained
  attention, via fine-tuning/distillation. This is explicitly a longer research thread per
  `AGENT_HANDOFF.md` §3 — track it, but don't let it block the hackathon deliverable.
- Whatever else the Week 1–4 work surfaces as a promising, not-yet-explored direction.
