# AGENT HANDOFF: MANGO-A100

**Version:** 2.0 — rewritten 2026-09-07 after Phase 1 completed. (v1.0 described a plan
built on assumptions that measurement has since overturned; do not work from memory of it.)
**Deadline:** Green Mind AI Hackathon 2026 (CMKL), starts ~2026-10-01.

---

## 0. If you are the agent on the GPU machine, read this section first

Phase 1 is **complete on the Apple M4 dev box**. Everything below is measured, with the raw
data committed in `results/`. Your job is Phase 1.5: **re-establish these results on CUDA**,
then continue to Phase 2.

**Start here, in order:**

```bash
python3 scripts/fetch_models.py          # 5.8 GB target model (gitignored, not in the repo)
./scripts/build_llama.sh                 # BACKEND=cuda auto-detected
export LLAMA_BIN_DIR=<path printed by the build script>
"$LLAMA_BIN_DIR/llama-cli" --help | grep -A2 -- --spec-type   # MUST list draft-mtp
python3 experiments/p14_energy/power.py auto                  # MUST print backend=nvml
python3 experiments/p13_bilingual/run_budget_sweep.py 1 2 3 5 8
```

Then compare your table against `results/P13c_FINDINGS.md` and record it as a new findings
file. **Expect the optimal `n_max` to be larger than the M4's** — see §4.

⚠️ **Repo layout matters — this project depends on two sibling repos by relative path.**
Cloning `mango-a100` alone will break immediately. The expected tree is:

```
<workspace>/
├── katgpt-rs/                      # github.com/katopz/katgpt-rs — the research crates
│   └── crates/katgpt-speculative/  #   referenced by experiments/e03_.../Cargo.toml
├── neural-engines/
│   ├── chimera/vendor/llama.cpp/   # the b10068 source with DFlash/MTP that Phase 1 built from
│   └── mango-a100/                 # ← this project
└── knowledge-base/                 # reusable research store, referenced by the docs
```

If your layout differs, override the two paths rather than editing files:
`LLAMA_SRC=/path/to/llama.cpp ./scripts/build_llama.sh`, and fix the `path = ...` line in
`experiments/e03_weaver_retarget_probe/Cargo.toml`. Upstream llama.cpp at b10068 or newer works
as a substitute for the chimera vendor copy — the requirement is only that
`llama-cli --help` lists `--spec-type` with `draft-mtp`.

**Confirmed hardware for this phase: NVIDIA RTX 4060 8 GB, Windows 11** (owner-confirmed
2026-09-07). Ada Lovelace `sm_89`, 272 GB/s, 8 GB VRAM. Consequences you must plan around:

| constraint | consequence |
|---|---|
| **8 GB VRAM, ~7 GB usable** (Windows reserves ~1 GB for display) | The Q4_K_M target model (5.77 GB) fits **but is tight.** Budget: weights 5.77 + GDN recurrent state ~0.20 (incl. 3 rollback snapshots) + KV cache 0.54 @16K ctx + CUDA buffers ~0.40 ≈ **6.9 GB**. Verify with `nvidia-smi` while loaded before trusting any run. |
| **Adding the DFlash drafter (1.13 GB) overflows** — ≈8.0 GB | **Use `--spec-type draft-mtp`** (the model's own MTP head, no second model) as the primary drafter here. `draft-dflash` needs either a smaller target quant (Q4_K_S 5.49 GB, or Q3_K_M) or partial CPU offload — and partial offload distorts the throughput numbers, so prefer changing quant. |
| **The AWQ W4A16 build (9.08 GB) does not fit at all** | The actual A100 deployment artifact cannot be validated here. Its acceptance rates may differ from Q4_K_M's — that check belongs to the A100 session, and must be listed as a known gap in any writeup. |
| **KV cache is unusually small** — only 8 of 32 layers cache K/V (the other 24 carry O(1) recurrent state) | ~34 KB/token, so 0.54 GB at 16K context. **Long context is limited by weights, not by KV.** Don't assume the usual KV-dominates intuition here. |
| **Ada has FP8 tensor cores; A100 (Ampere) does not** | See the warning below. This is the single most dangerous difference between this box and the target. |

Also note the FLOP/byte bracketing: M4 ≈ 38, **RTX 4060 ≈ 445**, A100 ≈ 153. The 4060 sits
*above* A100 on the compute-per-byte axis while the M4 sits far below, so the two dev boxes
**bracket the A100 from either side** — measure the optimal draft budget on both and the A100's
optimum should fall between them. Theoretical decode ceiling here: **50.7 tok/s** for a
5.37 GiB model (vs the M4's 22.3), so expect roughly 40–45 tok/s baseline at the ~80–85%
efficiency llama.cpp typically achieves.

**Never tune or benchmark an FP8 path for the A100 target — A100 (Ampere) has no FP8 tensor
cores.** This is confirmed, see `knowledge-base/topics/quantization.md`.

---

## 1. Mission

Optimize inference throughput/latency and energy efficiency of **MANGO1.5-Qwen3.5-9B** on
**A100**, preserving Thai-English bilingual quality — as a hackathon entry, and as a research
track that continues afterward. The owner prioritizes **novelty and long-term research value
over a marginal leaderboard win** (memory: `feedback-novelty-over-leaderboard`).

## 2. What the target model actually is (measured, E01/E04)

`cmkl/MANGO1.5-Qwen3.5-9B` — public on HF. `transformers` 4.57.6 cannot load it
(`model_type: qwen3_5` unknown); read its config raw via `huggingface_hub`.

- **24 of 32 layers are `linear_attention` (Gated DeltaNet); 8 are `full_attention`**,
  repeating `[L,L,L,F]` (`full_attention_interval = 4`).
- **Ships an MTP head** (`mtp_num_hidden_layers: 1`; in GGUF, `blk.32.nextn.*`,
  `qwen35.nextn_predict_layers = 1`). **No draft model needs training.**
- hidden 4096 · 32+1 blocks · vocab 248,320 · 256K context · dense (no MoE) · multimodal.
- SSM params: `conv_kernel 4, state_size 128, group_count 16, time_step_rank 32, inner_size 4096`.
- llama.cpp calls the architecture `qwen35` and supports it natively.

Published artifacts: BF16 original (19.3 GB) · **`CMKL/…-AWQ-W4A16-mtp` (9.08 GB, the
deployment build, MTP preserved)** · official GGUF set (Q4_K_M 5.78 GB — what Phase 1 used).

## 3. What Phase 1 measured (all on M4, all reproducible from `results/`)

| finding | file |
|---|---|
| Thai costs **1.845× more tokens** than English for the same content (chars/token 1.99 vs 4.46) | `P13a_FINDINGS.md` |
| MTP draft acceptance: **English 70.5% vs Thai 20.7% — a 3.41× gap** | `P13b_FINDINGS.md` |
| **Optimal draft budget is language-dependent.** At `n_max=1` English hits 21.49 tok/s, **+19.5% over baseline and 96% of the M4 bandwidth ceiling**. No Thai budget beats baseline — Thai's optimum is speculation **off**. llama.cpp's default `n_max=3` loses 18% (en) / 57% (th). | `P13c_FINDINGS.md` |
| `gdn_tree_verify` in katgpt-rs passes 16/16 tests (synthetic trees only — never yet run against the real model) | `E01_FINDINGS.md` §E02 |
| Weaver is shape-general: runs correctly at hidden=4096 (zero weights — plumbing only, not quality) | `E03_FINDINGS.md` |
| Energy harness built and math-verified; `nvml` backend **never executed on real hardware** | `P14_FINDINGS.md` |
| nextn/MTP C API is exported from libllama but **C++-mangled** — Phase 2 needs a ~20-line `extern "C"` shim | `P1_FINDINGS.md` |

## 4. What transfers to your machine and what does not

**Transfers (do not re-derive):** acceptance rates, the token-density ratio, the *shape* of
the finding that the optimum is language-dependent. These are properties of the model's
distributions and tokenizer, not of hardware.

**Does NOT transfer (must be re-measured):** every tok/s number, and **the value of the
optimal `n_max`**. That optimum tracks the machine's compute-to-bandwidth ratio:

| host | bandwidth | FP16 TFLOPS | **FLOP/byte** | decode ceiling, 5.37 GiB model |
|---|---:|---:|---:|---:|
| M4 base | 120 GB/s | 4.6 | **38** | 22.3 tok/s |
| **RTX 4060 8 GB ← this phase** | 272 GB/s | ~121 | **445** | 50.7 tok/s |
| A100 80 GB | 2039 GB/s | 312 | **153** | 379.7 tok/s |

The M4 (38) and the RTX 4060 (445) **bracket the A100 (153) from either side**. Measuring
the optimal budget at both ends lets you interpolate a prediction for A100 *before* spending
queue time — and "a hardware-aware model of the optimal draft budget, validated at three
points" is a contribution in its own right, not just tuning. **This is the main reason the
GPU machine matters more than more M4 runs.**

## 5. Architecture decision (D1) — unchanged, now with a known caveat

**Rust orchestrates; llama.cpp is the kernel library.** A Rust binary links llama.cpp via its
C API and calls katgpt-rs (`weaver`, `caddtree_budget`, `gdn_tree_verify`) as normal Rust
deps. Per-step FFI traffic is small (`h_verifier[4096]`, `dflash_logits[D][K]`).

**Caveat found in P1.2:** the nextn entry points (`llama_set_embeddings_nextn`,
`llama_get_embeddings_nextn_ith`, `llama_set_nextn_layer_offset`,
`llama_get_embeddings_layer_inp`) are exported but **C++-mangled** (they live in the
`llama-ext.h` staging API). Rust cannot bind them by name — write a small `extern "C"` shim
`.cpp` and link it. This is a build step, not a redesign.

## 6. Why speculation behaves badly here (mechanism, E-series + P1.3c)

24 of 32 layers carry recurrent state. llama.cpp rolls that state back after a rejected draft
using bounded per-token snapshots — `n_rs_seq`, observed = 3 at runtime, and marked
`[EXPERIMENTAL]` in `llama.h`. `LLM_ARCH_QWEN35` is on the supported list
(`llm_arch_supports_rs_rollback` → true). A pure-attention model pays nothing for a rejection
(it just truncates a KV index); **this model pays a recurrent-state restore every time**,
which is why Thai's low acceptance is so much more expensive than the same acceptance drop
would be elsewhere.

`katgpt-rs`'s `gdn_tree_verify` eliminates that cost by construction — its verify pass never
writes S₀, only `commit_accepted` does, along the single accepted path. **That is the
project's central technical bet, and P1.3c is the measured evidence of the problem it solves.**

## 7. Next work, in priority order

1. **Phase 1.5 (you, on the GPU box):** re-run the budget sweep; exercise the `nvml` energy
   backend for the first time; record a CUDA baseline; collect **≥3 repeats for error bars**
   (do this here, not on the M4 — these are the numbers that will be quoted).
2. **Widen the matrix:** `--spec-type draft-dflash` (drafter `z-lab/Qwen3.5-9B-DFlash`, GGUF
   mirror `lym00/Qwen3.5-9B-DFlash-GGUF-Test`, vocab 248320 = token-compatible with MANGO),
   plus the `ngram-*` types. Widen the prompt set too — 10 pairs is thin.
3. **Phase 2:** the `extern "C"` shim → Rust bin → `caddtree_budget` driving the draft budget
   per request. **P2.3 is the cleanest contribution and needs no training**: same model, same
   drafter, adaptive budget, measured against the fixed-budget baseline in `P13c_FINDINGS.md`.
4. **Prior-art search before any novelty claim** — see §8.
5. **A100:** one cheap early session (does it load, does NVML read, does the script finish)
   long before the real run. **Do not let competition day be the first contact with A100.**

## 8. Claim discipline — read before writing any pitch or paper text

- **We have made measurements, not inventions.** `gdn_tree_verify` implements Oda et al.
  (arXiv:2607.06763); CaDDTree is arXiv:2606.01813; BASTION is arXiv:2605.29727; MTP, DFlash,
  AWQ and llama.cpp are others' work. Cite them as such.
- Weaver *may* be original (it mirrors the owner's own `riir-train` Plan 314 and cites
  arXiv:2607.06763 only for the surrounding framework) — **unverified. Do a prior-art search
  before claiming it.**
- The bilingual acceptance-gap and language-dependent-budget results appear to be first of
  kind — **also unverified.** No "first to…" claim goes in any deliverable until searched.
- Every number carries its hardware. **M4 numbers are never presented as A100 numbers.**
- Report deltas against an identical baseline on identical hardware, never stack-vs-vLLM
  absolutes.
- Most "Nx speedup" figures in the source literature failed adversarial verification — see
  `../../knowledge-base/topics/` before quoting any of them.
- **CMKL already ships AWQ W4A16.** Quantization is the starting line, not a deliverable.
- Do **not** attribute to this project the Lily-style architecture description (MoE
  `fused_expert`, `folded_gqa`, ANE/CoreML offload, "204.8 tok/s vs vLLM"). Grep at katgpt-rs
  HEAD returns zero hits for those symbols; it describes a different system.

## 9. Where things live

- This project: `neural-engines/mango-a100/` — `PLAN.md` (v2), `APPROACH.md`, `PROGRESS.md`
  (the running log; **update it every session**), `results/` (all findings + raw JSON).
- Source to port from: `katgpt-rs/crates/{katgpt-speculative,katgpt-quant,katgpt-kv,katgpt-hla}`
  and `katgpt-core/src/gdn_tree_verify/` (feature-gated: `--features gdn_tree_verify`).
- llama.cpp source with DFlash/MTP: `neural-engines/chimera/vendor/llama.cpp` (b10068).
- Reusable research store: `../../knowledge-base/INDEX.md` — **check it before running any new
  deep research.**
- Models are gitignored: `scripts/fetch_models.py`.
