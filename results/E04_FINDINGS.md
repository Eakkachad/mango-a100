# E04 — MANGO1.5-Qwen3.5-9B runs on the M4, and the whole speculative stack lines up

**Date:** 2026-09-07 · **Hardware:** Apple M4, 16 GB, macOS
**Model:** `CMKL/MANGO1.5-Qwen3.5-9B-GGUF` → `MANGO1.5-Qwen3.5-9B-Q4_K_M.gguf` (5.78 GB on disk, 5.37 GiB loaded, 9.20B params)
**Runtime:** llama.cpp b9960 (`a935fbffe`, brew), Metal + BLAS backend

## The real target model runs on the dev box

`llama-bench -m …Q4_K_M.gguf -p 512 -n 128 -r 2`:

| test | t/s |
|---|---|
| **pp512** (prefill) | **214.73 ± 0.02** |
| **tg128** (generation) | **17.79 ± 0.03** |

Backend `BLAS,MTL` — Metal GPU is in use (`recommendedMaxWorkingSetSize` 12,713 MB).
An interactive `llama-cli` run answered a Thai prompt correctly and reported
`Prompt: 39.5 t/s | Generation: 18.8 t/s` (unwarmed, different config — the
`llama-bench` numbers above are the citable ones).

These are **M4 numbers and must never be presented as A100 numbers** — they exist to
prove the pipeline runs end-to-end and to give a relative baseline for algorithmic
work. Full log: `results/e04_llama_bench.log`.

⚠️ Operational note: `llama-cli` enters conversation mode and loops on EOF when stdin
is closed, appending `> ` forever — one run produced a **1 GB log** before being
killed. Always use `-st` / `--single-turn`, or `llama-bench`, for non-interactive runs.

## GGUF metadata confirms the architecture at tensor level

`general.architecture = qwen35` — llama.cpp has native support.

| key | value |
|---|---|
| `qwen35.block_count` | **33** (= 32 transformer layers + 1 nextn/MTP layer) |
| `qwen35.nextn_predict_layers` | **1** |
| `qwen35.full_attention_interval` | **4** (confirms the `[L,L,L,F]` pattern from E01) |
| `qwen35.embedding_length` | 4096 |
| `qwen35.attention.head_count` / `_kv` | 16 / 4 |
| `qwen35.attention.key_length` / `value_length` | 256 / 256 |
| `qwen35.ssm.conv_kernel` | 4 |
| `qwen35.ssm.state_size` | 128 |
| `qwen35.ssm.group_count` | 16 |
| `qwen35.ssm.time_step_rank` | 32 |
| `qwen35.ssm.inner_size` | 4096 |

442 tensors total. Layer composition is directly visible:

- **GDN layer (blk.0)**: `ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt.bias`,
  `attn_gate`, `attn_qkv` — a Gated DeltaNet / SSM block.
- **Full-attention layer (blk.3)**: `attn_q`, `attn_k`, `attn_v`, `attn_output`,
  `attn_q_norm`, `attn_k_norm` — standard attention with QK-norm.
- **MTP head (blk.32)**: `blk.32.nextn.eh_proj.weight`, `blk.32.nextn.enorm.weight`,
  `blk.32.nextn.hnorm.weight`, `blk.32.nextn.shared_head_norm.weight`.

## The decisive finding: every piece of the speculative stack already exists

llama.cpp treats the MTP layer as **excluded from the main forward pass** and exposes it
through a dedicated API — from the llama.cpp source vendored at
`neural-engines/chimera/vendor/llama.cpp` (**b10068**, 2026-07-18 — newer than the brew
b9960 binary):

- `llama-hparams`: `n_layer_nextn`, and `n_layer_eff() = n_layer_all - n_layer_nextn`.
- `llama-context.h`: `set_embeddings_nextn(bool, bool masked)`, `get_embeddings_nextn()`,
  `get_embeddings_nextn_ith(i)`, `set_nextn_layer_offset(offset)`.
- `common/speculative.cpp`: a **`COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH`** implementation —
  "DFlash: block-diffusion drafting with a draft-side KV cache injection" — calling
  `llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true)` and
  `llama_get_embeddings_nextn_ith(ctx_dft, i_batch)`. `tools/server/server-context.cpp`
  wires it into the server. The vendored HEAD commit is literally
  *"model: rotate injected K/V cache for DFlash (#25823)"*.

And on the katgpt-rs side, independently:

- `crates/katgpt-speculative/src/dflash.rs` — "DFlash — zero-alloc speculative
  marginal-distribution drafter", with an explicit **MTP-conditioning operation**.
- `crates/katgpt-speculative/src/weaver.rs` — takes **`dflash_logits` and `h_dflash` as
  its inputs** and computes `corrected = dflash_logits + weaver_residual`.
- `crates/katgpt-speculative/src/caddtree_budget.rs` — adaptive draft-tree budget.
- `crates/katgpt-core/src/gdn_tree_verify/` — rollback-free tree verification against
  GDN layers (16/16 tests green on this machine, E02).

**These were built against the same drafting method (DFlash) and the same head (MTP)
that the target model ships and that llama.cpp already implements.** The composition is:

```
MTP/nextn head (in the model)  →  DFlash drafting (llama.cpp + katgpt-rs)
      →  Weaver residual correction of the draft logits (katgpt-rs)
      →  CaDDTree adaptive tree budget (katgpt-rs)
      →  rollback-free GDN tree verification (katgpt-rs)  →  commit
```

No piece of that chain has to be invented — the work is integration plus measurement.

## Open questions this raises

1. Does the **brew b9960** binary include the DFlash/MTP speculative path, or only the
   newer vendored b10068 source? (`strings` on the brew binary found no `nextn`/`dflash`
   symbols, but that is weak evidence.) If not, build llama.cpp from the vendored source.
2. Is there a **DFlash draft model published for MANGO/Qwen3.5**, or does the MTP head
   alone serve as the drafter? `common/speculative.cpp` asserts DFlash "requires ctx_tgt
   and ctx_dft" and reads a `dflash.block_size` metadata key — implying a separate draft
   GGUF is expected.
3. What is the **baseline MTP/DFlash acceptance rate** on MANGO — and how much does
   Weaver correction plus CaDDTree budgeting move it? That is thecore number for the whole
   project, and it transfers from M4 to A100 unchanged.
