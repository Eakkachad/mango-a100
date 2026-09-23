# Research References

> **Canonical copy has moved.** The full, machine-searchable, cross-linked version of everything below —
> plus a lot of raw unverified data this file doesn't include — now lives in
> [`../../knowledge-base/`](../../knowledge-base/INDEX.md), organized by topic and reusable for any future
> task, not just this project. **Update the knowledge base first, then sync this file** if this project's
> condensed view needs to change; don't let the two drift apart.

Condensed local copy of both research passes behind this project, so the reasoning survives even if the
published artifact link ever goes stale. Full external report with all sources:
**[The MANGO Inference Playbook](https://claude.ai/code/artifact/6b913516-9c95-42bd-ae2f-b8ab7f2afea8)**
(111-agent deep-research pass, 28 sources fetched, 132 claims extracted, 25 claims put through 3-vote
adversarial verification: 17 confirmed, 8 refuted).

---

## Part A — External literature: confirmed findings

| Finding | Confidence | Source |
|---|---|---|
| **A100 has no native FP8 tensor cores.** NVIDIA's own TensorRT-LLM FP8 benchmarks/guidance are explicitly built for and demonstrated on Hopper/Ada, not Ampere. → use INT4/INT8 on A100, not FP8. | medium | [TensorRT-LLM quantization blog](https://nvidia.github.io/TensorRT-LLM/blogs/quantization-in-TRT-LLM.html) |
| **AWQ and SmoothQuant are the practical training-free INT4/INT8 candidates.** AWQ protects the top ~1% salient weight channels (via activation statistics, not weight magnitude); SmoothQuant migrates quantization difficulty from activations to weights, training-free, W8A8. Both calibration-light, no fine-tuning needed. *Their headline "3x"/"1.56x" speedup figures did NOT survive verification — only the mechanism claims did.* | high | [AWQ, MLSys 2024](https://arxiv.org/abs/2306.00978), [SmoothQuant, ICML 2023](https://arxiv.org/pdf/2211.10438) |
| **QServe shows a tuned INT4/INT8 pipeline's ceiling on A100.** Generic INT4-weight kernels lose 20–90% of their gain to runtime dequant overhead. QServe's W4A8KV4 (4-bit weights, 8-bit activations, 4-bit KV cache + SmoothAttention) gets 1.2x (Llama-3-8B) / 2.4x (Qwen1.5-72B) vs. TensorRT-LLM, measured on A100. | high | [QServe, MLSys 2025](https://arxiv.org/pdf/2405.04532) |
| **No single quantization method wins on latency, energy, and quality simultaneously.** 11-config study on Llama-2/CodeLlama: quantization improves energy/token up to 30%, but 8-bit activation gives minimal energy savings despite lower latency — the real energy win is 4-bit weights (W4A8KV4 best). Quality can drop severely on smaller/harder-task models (up to 92% HumanEval drop, 13B model). **A100 favors energy efficiency at moderate load; H100 favors latency/scalability.** Measured on Llama/CodeLlama, not Qwen3.5 — magnitude transfer unverified. | medium | [Characterizing quantization tradeoffs, ASPLOS 2026](https://arxiv.org/pdf/2508.16712) |
| **Speculative decoding is lossless and roughly doubles throughput.** Decoding is memory-bandwidth-bound (streaming all params from HBM per step); spec-decoding amortizes that transfer over multiple verified tokens per pass. EAGLE: theoretically guaranteed distribution-preserving, no target-model fine-tuning, draft head trains in 1–2 days on 4×A100. Medusa-1: heads train in ~5 hours on 1×A100. *Medusa's own "2.2–3.6x" headline figure did NOT survive verification — only the training-cost claim did.* All figures are Llama-family (7B–70B), not yet confirmed on Qwen3.5-9B. | medium | [EAGLE, ICML 2024](https://arxiv.org/pdf/2401.15077), [Medusa, ICML 2024](https://arxiv.org/pdf/2401.10774) |
| **SGLang's RadixAttention gives free KV-cache reuse across shared-prefix requests**, zero custom code, just from choosing the engine. *The paper's own "6.4x vs. SOTA" headline did NOT survive verification — only the mechanism claim did.* | high | [SGLang, NeurIPS 2024](https://arxiv.org/abs/2312.07104) |

## Part B — Refuted (don't cite these in the hackathon writeup without re-benchmarking yourself)

| Claim | Vote | Source |
|---|---|---|
| SGLang: up to 6.4x higher throughput vs. SOTA systems | 1–2 | arxiv.org/abs/2312.07104 |
| AWQ/TinyChat: >3x speedup over HF FP16 on GPUs | 1–2 | arxiv.org/abs/2306.00978 |
| GPTQ: "negligible" accuracy degradation at 3–4 bits | 0–3 | arxiv.org/abs/2210.17323 |
| GPTQ: ~3.25x end-to-end speedup on A100 specifically | 0–3 | arxiv.org/abs/2210.17323 |
| SmoothQuant: up to 1.56x speedup, 2x memory reduction, negligible accuracy loss | 1–2 | arxiv.org/pdf/2211.10438 |
| FP8 @ batch 16 under 500ms TTFT constraint: 2.3x speedup vs FP16 | 0–3 | nvidia.github.io/TensorRT-LLM |
| Medusa-1: >2.2x speedup no quality loss; Medusa-2: 2.3–3.6x | 1–2 | arxiv.org/pdf/2401.10774 |
| SpecDec (non-AR draft Transformer): ~5x speedup at comparable quality | 1–2 | arxiv.org/pdf/2401.07851 |

**Pattern:** nearly every precise "Nx speedup" marketing-style figure failed the 3-vote check; qualitative
mechanism claims and a few narrowly-scoped benchmark tables (QServe's 1.2x/2.4x, TensorRT-LLM's own
H100 1.51x/1.40x) held up. Re-benchmark any multiplier on your own MANGO1.5-Qwen3.5-9B + A100 setup before
using it in the pitch.

## Part C — Open questions (unresearched, not disproven)

1. What NVML sampling cadence / CodeCarbon integration / Prometheus GPU exporter setup is actually best
   for tokens-per-joule measurement? No claim on this survived verification — build and validate directly.
2. What quantified, A100-specific guidance exists for continuous batching, tensor-parallel degree, CUDA
   graph capture? Same gap.
3. Are AWQ/SmoothQuant/GPTQ/QServe kernels and checkpoint formats compatible out-of-the-box with the
   Qwen3.5 architecture, or does MANGO1.5-Qwen3.5-9B need adaptation? All verified benchmarks above are
   Llama-family.
4. Does a pretrained EAGLE/Medusa-style draft head already exist for a Qwen3.5-class model? Would skip the
   training step entirely.

---

## Part D — Local component map (`katgpt-rs` and workspace)

**Bottom line from the local audit:** nothing in the workspace runs on NVIDIA hardware today — it's a
from-scratch Apple Silicon (CPU/Metal/ANE) research program. Two components are worth porting; everything
else should be left alone.

| Crate / path | What it is | Port status |
|---|---|---|
| `katgpt-rs/crates/katgpt-speculative` | **Primary novelty bet.** `weaver.rs` — a correction/verification module validated against a **real Gemma2-2B checkpoint** (`tests/weaver_real_checkpoint.rs`, gated behind `weaver_runtime` feature + external `WEAVER_CHECKPOINT_PATH`, ~219MB, lives outside this repo). `caddtree_budget.rs` — cost-aware adaptive draft-tree budget selection, grounded in CaDDTree (arXiv:2606.01813) and BASTION (arXiv:2605.29727), both 2026 papers. Also: `verifier_trait.rs`, `spec_generator.rs`, `belief_drafter.rs`, `prefix_scheduler.rs`, `domino_lora.rs`. **Needs a dedicated deep-dive (Week 1) — only discovered, not fully audited.** |
| `katgpt-rs/crates/katgpt-quant` | 5 KV-cache vector-quantization codecs (TurboQuant, PlanarQuant, IsoQuant, OCTOPUS, Hybrid OCT-PQ), single-file crate. Compresses KV *values*, not weights. Near-term, low-risk port target — no retraining needed, bolts onto any attention implementation. | Port candidate, Week 2. |
| `katgpt-rs/crates/katgpt-kv` | `PagedKVCache` (copy-on-write across speculative-decoding candidate branches — for tree exploration, not vLLM-style multi-request memory paging), plus `async_qdq.rs`, `osc_kv.rs`, `kv_share.rs`, `targeted_precision.rs`. | Relevant alongside `katgpt-speculative`'s tree logic. |
| `katgpt-rs/crates/katgpt-hla` | Higher-order Linear Attention — custom O(1)-cache linear-attention kernel (`kernel.rs`), validated only on a toy 27-vocab, 1-layer model. Architecturally different from the standard softmax attention MANGO1.5-Qwen3.5-9B was trained with. | **Stretch/post-hackathon research track**, not a committed deliverable — needs distillation/fine-tuning to swap in without quality loss. |
| `katgpt-rs/crates/katgpt-attn-match/src/score_matrix_gpu.rs` | Explicit "GPU dispatch stub" — wires a Metal dispatch path with automatic fallback to CPU Rayon when no GPU kernel is compiled in. No working GPU kernel yet, on any backend. | Good reference for how to *structure* a dispatch-with-fallback path; not reusable CUDA code. |
| `neural-engines/chimera` | Rust engine for Qwen2.5 (0.5B/1.5B/7B), F16/Q4, cross-model latent alignment. Has the workspace's only real CUDA code: a hand-rolled cuBLAS GEMV kernel for the LM head only, ~243 tok/s on RTX 4060. | Tangential — one kernel, not a full stack, RTX 4060 numbers only. |
| `neural-engines/research_workspace/katcompress` | Small PyTorch script: per-layer SVD condition numbers on real HF Qwen2.5-1.5B weights, flags which layers tolerate aggressive quantization. | **Reusable** — repoint at MANGO1.5-Qwen3.5-9B, use as an AWQ group-size/precision selector. |
| `neural-engines/omlx` (third-party clone, not user's code) | MLX-based local server: continuous batching, tiered PagedAttention-style KV cache, explicitly supports Qwen3.5. | Design reference only — Apple Silicon/MLX, not portable to CUDA. |
| Everything else (`dg-sdlm`, `geometric-sheaf-llm`, `A.L.I.C.E`, `colibri`, `nova-fusion`, `AXIOM`, `Bonsai-demo`, `kimi-local-engine`, `gargantua-sheaf`) | Separate research directions (diffusion LMs, sheaf-theoretic attention, agent cognition, MoE disk-streaming, an unrelated NASA hackathon pipeline). | Out of scope. |

No NVML/CodeCarbon/power-measurement code exists anywhere in the workspace — that has to be built fresh
regardless (see Open Question 1 above).
