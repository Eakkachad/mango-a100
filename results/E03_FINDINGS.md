# E03 — Weaver retarget probe + model-availability survey

**Date:** 2026-09-07 · **Repro:** `cd experiments/e03_weaver_retarget_probe && cargo run --release`

## E03a — Weaver is shape-general: retarget to MANGO (hidden 4096) is viable

`katgpt-speculative`'s Weaver corrector was only ever exercised at Gemma2-2B's shape
(hidden 2304), which raised the question of whether it was architecture-locked. It
isn't: `WeaverConfig` takes `hidden_dim` / `n_heads` / `d_ff` / `max_depth` /
`k_candidates` as plain fields, and `WeaverInput` derives dimensions from slice
lengths. Ran the same code path at four shapes with `WeaverWeights::zeros`:

| shape | hidden | heads | head_dim | D | K | V | correct() | result |
|---|---|---|---|---|---|---|---|---|
| Gemma2-2B (control) | 2304 | 16 | 144 | 4 | 32 | 911 | 65.0 ms | PASS |
| paper default (control) | 2048 | 16 | 128 | 8 | 32 | 1024 | 33.2 ms | PASS |
| **MANGO-9B** | **4096** | 16 | 256 | 4 | 32 | 1024 | 154.2 ms | **PASS** |
| **MANGO, K=128** | **4096** | 16 | 256 | 8 | 128 | 4096 | 131.5 ms | **PASS** |

Invariants checked at every shape: output shapes match `(D, K)`; corrected
probabilities are finite, non-negative, and sum to 1 within 1e-3 per depth;
corrected logits finite; and zero weights produce a ~0 residual (the documented
no-op contract).

**Scope of this result:** zero weights prove *shape/plumbing* generality only, not
numerical quality. A MANGO-shaped trained Weaver checkpoint is still needed for real
acceptance-rate numbers. The timings above are single-shot, unwarmed, f32-on-CPU —
**not** benchmarks; `benches/bench_136_weaver_f16_latency.rs` is the real harness.

Feature gate: `katgpt-speculative/weaver_runtime` (module compiles out without it).

## E03b — What CMKL has already published (changes the plan)

| Repo | Size | What it is |
|---|---|---|
| `cmkl/MANGO1.5-Qwen3.5-9B` | 19.3 GB | BF16 original, 5 shards |
| **`CMKL/MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp`** | **9.08 GB** | **Official AWQ W4A16 — 4-bit, group_size 128, `pack-quantized` (compressed-tensors), MTP head preserved** |
| `CMKL/MANGO1.5-Qwen3.5-9B-GGUF` | 59 GB total | Official GGUF set: BF16 / Q8_0 / Q6_K / Q5_K_M / **Q4_K_M 5.78 GB** / Q4_K_S |
| `mlx-community/Qwen3.5-9B-MLX-4bit` | 5.98 GB | Base Qwen3.5-9B, MLX 4-bit (group 64), MTP preserved — *base model, not MANGO* |

**Strategic consequence: AWQ quantization is the starting line, not our contribution.**
`PLAN.md` Week 4 lists "integrate AWQ/SmoothQuant as the safety net" as a deliverable —
CMKL already ships exactly that, quantized W4A16 *with MTP intact*. Re-doing it wins
nothing. The differentiator has to be the speculative-decoding / GDN work.

## E03c — Runtime availability on the M4 dev box

- **llama.cpp b9960 (`a935fbffe`) installed via brew** — `llama-cli`, `llama-server`,
  `llama-bench`, `llama-batched-bench`, and speculative-decoding flags
  (`--spec-draft-*`, `--hf-repo-draft`) are present. Combined with CMKL's official
  GGUF, this is the most direct path to running **the real target model** on the M4.
- **mlx-lm 0.29.1 does NOT support `qwen3_5`** (`mlx_lm.models.qwen3_5` missing). It
  does support `qwen3_next` and `qwen3`. So the MLX 4-bit route is blocked for now
  unless mlx-lm adds the architecture (worth re-checking later — `qwen3_next` support
  means the GDN-hybrid family is already modelled there).
- `transformers` 4.57.6 also cannot load `qwen3_5` via `AutoConfig` (see E01).

**E04 therefore runs on llama.cpp + `MANGO1.5-Qwen3.5-9B-Q4_K_M.gguf` (5.78 GB, fits
16 GB comfortably).** Download started; see `results/e04_download.log`.
