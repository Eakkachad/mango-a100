# E01 — Target Architecture Probe: MANGO1.5-Qwen3.5-9B

**Date:** 2026-09-07 · **Status:** measured, reproducible
**Repro:** `python3 experiments/e01_target_arch_probe/fetch_raw_config.py`
(raw config saved to `results/e01_raw_configs.json`)

## The gating question is answered: YES, the target model is a hybrid GDN model.

`cmkl/MANGO1.5-Qwen3.5-9B` is public on HuggingFace. `transformers` 4.57.6 cannot
load it via `AutoConfig` (`model_type: qwen3_5` unrecognized — config declares
`transformers_version: 4.57.0.dev0`), so the config was fetched raw via
`huggingface_hub.hf_hub_download`. Its config is **identical** to `Qwen/Qwen3.5-9B`
on every architectural field below.

| Field | Value |
|---|---|
| `model_type` | `qwen3_5` (text: `qwen3_5_text`) |
| `hidden_size` | 4096 |
| `num_hidden_layers` | 32 |
| **`layer_types`** | **24 × `linear_attention` + 8 × `full_attention`** — repeating `[L, L, L, F]` |
| full-attn heads / KV heads | 16 / 4 (GQA 4:1) |
| linear key / value heads | 16 / 32, head_dim 128 / 128, `linear_conv_kernel_dim` 4 |
| `mamba_ssm_dtype` | `float32` |
| **`mtp_num_hidden_layers`** | **1** (multi-token prediction head, `mtp_use_dedicated_embeddings: false`) |
| `vocab_size` | 248,320 |
| `max_position_embeddings` | 262,144 (256K) |
| `intermediate_size` | 12,288 |
| MoE | none (dense) |
| vision | present (`vision_config`, depth 27) — model is multimodal |

## Why this decides the project

1. **`katgpt-rs`'s `gdn_tree_verify` applies directly to 75% of the model.**
   `crates/katgpt-core/src/gdn_tree_verify/` implements rollback-free speculative
   *tree* verification against Gated DeltaNet recurrent layers (arXiv:2607.06763
   §3.4, Oda et al.). The target has 24 such layers. Verified working on this
   machine — see E02 below.

2. **The model ships its own draft head.** `mtp_num_hidden_layers: 1` means
   multi-token prediction is built in — this closes the open question "does a
   pretrained EAGLE/Medusa-style draft head exist for a Qwen3.5-class model?"
   **We do not need to train a draft model at all.** MTP proposes → DDTree/CaDDTree
   shapes and budgets the draft tree → `gdn_tree_verify` verifies it against the
   GDN layers without rolling back recurrent state. All three pieces already exist.

3. **KV-cache quantization drops in priority.** Only 8 of 32 layers keep a KV cache
   (the other 24 carry O(1) recurrent state instead), so KV-cache memory pressure is
   ~1/4 of a conventional 32-layer model's. `katgpt-quant`'s codecs are still usable
   but the payoff is much smaller than assumed in `PLAN.md` Week 2 — **the GDN
   recurrent-state path is where the value moved.** Revisit that plan.

4. **The LM head is huge.** 4096 × 248,320 ≈ 1.02B parameters in the output embedding
   alone (~11% of the model). Quantizing/handling the LM head is disproportionately
   valuable here — note `tie_word_embeddings: false`, so input and output embeddings
   are separate.

5. **256K context** is where a hybrid GDN model's O(1)-state layers pay off most —
   the long-context regime is the strongest story for both benchmarks and a paper.

## E02 — katgpt-rs GDN tree verification runs green on this machine

`cargo test -p katgpt-core --features gdn_tree_verify --lib gdn_tree_verify::`
→ **16 passed, 0 failed** (Apple M4, 16GB, Rust 1.97.1, ~6.5s build).

Note the module is **feature-gated** (`--features gdn_tree_verify`); without the flag
it silently compiles out and the test filter matches nothing.

Tests passing include the ones that matter for correctness claims:
`test_random_trees_correctness`, `test_multihead_matches_reference`,
`test_linear_chain_matches_sequential`, `test_branching_tree_matches_per_branch`,
`test_commit_path_matches_sequential`, `test_verify_alloc_free_hot_path`, and three
`test_topology_from_ddtree_*` integration tests binding it to the DDTree structure.

## Correction to earlier project assumptions

An architecture description circulating for `katgpt-rs` (Lily-style specialized engine,
MoE `fused_expert`/`routed_moe` path, `folded_gqa`, `q4_gemm`, ANE/CoreML hybrid
offload via `omlx`, 204.8 tok/s vs vLLM on M3 Ultra) **does not correspond to this
repository's code.** Grep at HEAD `c478ab9f` returns zero files for
`deltanet_scan`, `folded_gqa`, `q4_gemm`, `grouped_gemm`, `fused_expert`, `routed_moe`,
`moe_routing`, `step_verify`, `hybrid_split`, `fixed_prefill_shape`, `prefix_cache`,
`kv_cache_lookup`, `omlx`, `Lily`; there are no `.metal`/`.cu`/`.ptx` files anywhere;
and `CoreML`/`ANE` appear only in a single aspirational comment
(`crates/katgpt-forward/src/forward.rs:138`). Those benchmark numbers belong to a
different system and must not be attributed to this work.
