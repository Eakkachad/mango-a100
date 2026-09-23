# Approach & Engineering Guidelines

## Strategy in one paragraph

Build MANGO1.5-Qwen3.5-9B's serving path on **Candle** (Rust, CUDA backend) rather than hand-rolling CUDA
kernels from scratch or defaulting to a vLLM-only build. Candle supplies the boring 90% — GEMM, standard
attention, safetensors loading, an existing CUDA compilation path — for free, so the month's engineering
goes entirely into the genuinely novel `katgpt-rs` components: the `katgpt-speculative` Weaver/DDTree
system and the `katgpt-quant` KV-cache codecs. This is a deliberate trade of "might not beat vLLM's raw
number" for "keeps a real, continuable research contribution," per the user's stated priority
(`[[feedback-novelty-over-leaderboard]]`).

## Component priority & risk tiers

1. **Foundation (must-have): Candle + CUDA baseline.** MANGO1.5-Qwen3.5-9B loading and generating
   correctly on both 4060 (INT4) and A100 (can also run FP16/INT8 there, more headroom). This is the
   reference point everything else is measured against — build it, and the benchmark harness, *before*
   any optimization.
2. **Near-term differentiator (low risk): `katgpt-quant` KV-cache codecs ported to CUDA kernels.** Bolts
   onto an already-correct forward pass. No retraining risk. Validate compression ratio vs. perplexity
   delta on the 4060 first.
3. **Primary differentiator (medium risk, high payoff): `katgpt-speculative`'s Weaver/DDTree system,
   retargeted from Gemma2-2B to Qwen3.5-9B.** This is the strongest novelty bet in the workspace — it's
   already literature-current (CaDDTree, BASTION, both 2026) and already validated against a real
   checkpoint, just not this model family. Week 1 deep-dive determines feasibility before committing
   further weeks to it.
4. **Safety-net baseline (low risk, standard technique, not from `katgpt-rs`): AWQ or SmoothQuant weight
   quantization.** If the above two run behind schedule, this alone — served through Candle or, as a
   fallback, vLLM — still delivers a defensible, literature-backed submission.
5. **Stretch / explicitly post-hackathon: HLA distillation.** Attempt only after 1–4 are solid, and only
   as a research experiment reported honestly as in-progress, not as a claimed hackathon result.

## Guidelines

- **Correctness before speed, every time.** Never benchmark a kernel whose output hasn't been checked
  against a reference (exact match for dequant/logic ops; perplexity-delta bound for lossy quantization).
- **No FP8 on the A100 path.** Keep any FP8 code path (useful for 4060-side experiments) behind a feature
  flag that's off by default; never let the A100 submission path depend on it.
- **Build per-architecture kernel targets from day one**: `sm_89` (4060) and `sm_80` (A100) as separate
  compilation targets, not a single "portable" launch config assumed to be optimal on both. Occupancy
  tuning (block/grid sizes) must be redone per architecture — see `AGENT_HANDOFF.md` §5.
- **One benchmark harness, same code, both GPUs.** Log: tokens/sec, TTFT, inter-token latency, perplexity
  delta vs. an FP16/BF16 reference, and — A100 sessions only — GPU power draw via NVML/nvidia-smi
  (`--query-gpu=power.draw`) and/or CodeCarbon, reported as tokens-per-joule. Don't build two different
  harnesses for the two GPUs; the whole point is that the *harness* is portable even though the *numbers*
  aren't.
- **A100 queue discipline.** Every queued session should run a pre-written, non-interactive script.
  Schedule a few cheap early sessions to sanity-check direction (still memory-bound there? still the same
  sign of effect?) rather than one big validation burn saved for the end.
- **Don't quote a literature multiplier without re-measuring it.** See `RESEARCH_REFERENCES.md` Part B —
  most precise "Nx speedup" figures in the source papers failed adversarial verification.
- **Keep this doc set current.** Update `PROGRESS.md` every working session; update `AGENT_HANDOFF.md`
  §3/§6 if a key decision changes, so the next agent (or the next you, a week later) isn't working from a
  stale plan.

## Definition of done (hackathon submission)

- MANGO1.5-Qwen3.5-9B serving on A100 via the Candle-based engine, with at minimum:
  - AWQ or SmoothQuant weight quantization (INT4/INT8) active.
  - KV-cache quantization (ported `katgpt-quant` codec) active.
  - Measured, on real A100 hardware (not extrapolated from the 4060): tokens/sec, TTFT, inter-token
    latency, average and peak GPU power draw, tokens-per-joule, and perplexity delta vs. an unquantized
    baseline.
- Stretch, in order of ambition: speculative decoding via the retargeted Weaver/DDTree system;
  quality-preserving results reported from the HLA distillation experiment (even partial/negative results
  are honest research output, per the standard this workspace already holds itself to — see
  `dg-sdlm/BENCHMARK_REPORT.md`'s correction notice for the norm to follow: report what's real, not what's
  hoped-for).
