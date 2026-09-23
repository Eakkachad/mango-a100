# Progress Log

Living log. Append a dated entry each working session — don't rewrite history, add to it. Keep the
**Status board** at the top current; it's the fast-scan summary for anyone (including a fresh agent)
picking this up.

---

## Status board

| Phase | State | Last touched |
|---|---|---|
| Phase 1 (plan v2) — novel result without integration | **COMPLETE** — P1.1–P1.4 all done | 2026-09-07 |
| Week 2 — KV-cache quantization port | not started | — |
| Week 3 — Speculative decoding | not started | — |
| Week 4 — Integration + A100 validation | not started | — |
| Post-hackathon — HLA distillation | not started | — |

**A100 queue sessions used so far:** 0.  ·  **M4 baseline (Q4_K_M, llama.cpp Metal): pp512 214.7 t/s / tg128 17.8 t/s**

---

## Log

### 2026-09-02 — Planning complete, scaffold created

- Ran a deep-research pass (111 agents, 28 sources, 3-vote verification) on LLM inference optimization
  for A100 — published as [The MANGO Inference Playbook](https://claude.ai/code/artifact/6b913516-9c95-42bd-ae2f-b8ab7f2afea8).
- Audited the local workspace (`katgpt-rs` + every `neural-engines/*` project) for reusability. Initial
  pass concluded "nothing ports directly, build fresh on vLLM"; user pushed back, correctly, once the real
  timeline (1 month, not hackathon-day hours) was on the table, and stated a preference for novelty /
  continuing-research value over a marginal leaderboard win.
- Re-scoped the strategy around porting `katgpt-rs`'s novel components onto a Candle (Rust+CUDA)
  substrate instead of a full from-scratch CUDA build or a vLLM-only build.
- Discovered mid-planning that `katgpt-speculative` is more mature than the first audit conveyed — it has
  a Weaver corrector validated against a real Gemma2-2B checkpoint, and a literature-current
  (CaDDTree/BASTION, both 2026) adaptive tree-budget selector. This became the primary novelty bet,
  replacing an earlier assumption of building EAGLE/Medusa from scratch.
- Created this folder (`neural-engines/mango-a100/`) with `README.md`, `AGENT_HANDOFF.md`,
  `RESEARCH_REFERENCES.md`, `APPROACH.md`, `PLAN.md`, and this file, so the plan survives across sessions.
- Saved two memory entries: `hackathon-green-mind-2026` (project context) and
  `feedback-novelty-over-leaderboard` (user preference, for future recommendations generally).
- **No code has been written.** Next action: Week 1, deep-dive `katgpt-speculative`.


### 2026-09-07 — E01/E02 measured: the gating question came back GO

**MANGO1.5-Qwen3.5-9B is a hybrid Gated-DeltaNet model — 24 of its 32 layers are
`linear_attention`, 8 are `full_attention`** (repeating `[L,L,L,F]`), hidden 4096,
256K context, vocab 248,320, dense (no MoE), multimodal, **and it ships an MTP
(multi-token prediction) head** (`mtp_num_hidden_layers: 1`). Full detail and repro:
[`results/E01_FINDINGS.md`](./results/E01_FINDINGS.md), raw config in
`results/e01_raw_configs.json`.

Consequences, in order of importance:

1. **`gdn_tree_verify` applies directly to 75% of the target model.** The
   rollback-free GDN tree verification in `katgpt-rs`
   (`crates/katgpt-core/src/gdn_tree_verify/`, arXiv:2607.06763) is exactly the
   algorithm this model's architecture needs for speculative decoding. This was the
   open question that decided whether the whole katgpt-rs bet was viable — it's a yes.
2. **No draft model needs training.** The model's built-in MTP head is the drafter.
   The open question "does a pretrained draft head exist for a Qwen3.5-class model"
   is closed — Week 3's training budget is freed.
3. **KV-quant priority drops.** Only 8/32 layers hold a KV cache, so `katgpt-quant`'s
   codecs address ~1/4 the memory pressure originally assumed. `PLAN.md` Week 2 should
   be re-weighted toward the GDN recurrent-state path.
4. **`cargo test -p katgpt-core --features gdn_tree_verify --lib gdn_tree_verify::`
   → 16 passed / 0 failed** on the M4 (the module is feature-gated; without the flag
   the tests silently filter out to zero).

Also recorded in E01_FINDINGS.md: an architecture description circulating for
katgpt-rs (MoE fused-expert path, folded GQA, ANE/CoreML offload, 204.8 tok/s vs vLLM)
**does not match this repo's code** — zero grep hits for those symbols, no GPU kernel
files at all. Those numbers belong to a different system; don't cite them as ours.

**Next:** E03 Weaver retarget probe (WeaverInput is config-driven — `hidden_dim` comes
from slice length, Gemma's 2304 only appears in a test fixture, so retargeting to
hidden=4096 looks viable), then E04 acceptance-rate harness.


### 2026-09-07 (cont.) — E03 measured: Weaver retargets cleanly; CMKL already shipped AWQ

- **E03a: Weaver is shape-general.** Ran `WeaverCorrector::correct` at four shapes
  (Gemma2-2B 2304 control, paper 2048 control, MANGO 4096 at D=4/K=32 and D=8/K=128)
  with zero weights — **all four PASS** on shape, finiteness, probability-sum, and the
  zero-residual no-op contract. Retargeting Weaver to hidden=4096 is viable. Probe:
  `experiments/e03_weaver_retarget_probe/` (standalone cargo bin, depends on
  `katgpt-speculative` with `weaver_runtime`). Caveat: zero weights prove plumbing, not
  numerical quality — a MANGO-shaped trained checkpoint is still needed.
- **E03b: CMKL already publishes `MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp`** (9.08 GB, 4-bit
  group_size 128, compressed-tensors `pack-quantized`, **MTP head preserved**), plus a
  full official GGUF set (Q4_K_M 5.78 GB). **This kills AWQ as a deliverable** — `PLAN.md`
  Week 4's "AWQ safety net" is already provided by the organizers. Differentiation must
  come from the speculative-decoding / GDN side.
- **E03c: runtime on the M4** — llama.cpp b9960 is installed with speculative flags;
  mlx-lm 0.29.1 does **not** support `qwen3_5` (does support `qwen3_next`); transformers
  4.57.6 can't load it either. So E04 runs on llama.cpp + the official MANGO GGUF.

Full detail: [`results/E03_FINDINGS.md`](./results/E03_FINDINGS.md).

**Next:** E04 — download of `MANGO1.5-Qwen3.5-9B-Q4_K_M.gguf` in flight; then measure
MTP draft acceptance rate on the real target model (the number that transfers to A100
unchanged), and check whether llama.cpp exposes the MTP head or whether we need our own
draft loop.

**PLAN.md needs re-weighting** (not yet edited): Week 2 KV-quant down (only 8/32 layers
cache KV), Week 4 AWQ out (already shipped by CMKL), GDN/MTP speculative path up.


### 2026-09-07 (cont.) — E04: the real model runs on the M4, and the stack composes

- **MANGO1.5-Qwen3.5-9B Q4_K_M runs on the M4 via llama.cpp b9960 (Metal).**
  `llama-bench`: **pp512 214.73 ± 0.02 t/s, tg128 17.79 ± 0.03 t/s** (5.37 GiB loaded,
  9.20B params). Answers Thai prompts correctly. M4 numbers — never cite as A100.
- **GGUF confirms the architecture at tensor level**: `general.architecture = qwen35`,
  `block_count 33` (32 + 1 nextn), `nextn_predict_layers 1`, `full_attention_interval 4`,
  `ssm.{conv_kernel 4, state_size 128, group_count 16, time_step_rank 32, inner_size 4096}`.
  MTP head present as `blk.32.nextn.{eh_proj,enorm,hnorm,shared_head_norm}.weight`.
  blk.0 carries SSM/GDN tensors, blk.3 carries full-attention tensors — exactly the
  `[L,L,L,F]` pattern.
- **The decisive finding:** llama.cpp already implements DFlash drafting on the MTP head
  (`COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH`, `llama_set/get_embeddings_nextn`) — and
  katgpt-rs's `dflash.rs` / `weaver.rs` were built against **that same drafting method**
  (`WeaverInput` literally takes `dflash_logits` + `h_dflash`). So the chain
  **MTP → DFlash → Weaver correction → CaDDTree budget → GDN rollback-free tree verify**
  needs integration and measurement, not invention. Note the llama.cpp vendored in
  `neural-engines/chimera/vendor/llama.cpp` is **b10068 — newer than the brew b9960** and
  its HEAD commit is literally a DFlash KV-cache fix.
- Detail: [`results/E04_FINDINGS.md`](./results/E04_FINDINGS.md).

⚠️ Operational trap recorded: `llama-cli` loops on EOF in conversation mode and wrote a
**1 GB log** before being killed. Use `-st`/`--single-turn` or `llama-bench`.

**Next:** (1) confirm whether brew b9960 has the DFlash path or we must build from the
vendored b10068 source; (2) find or produce a DFlash draft model for MANGO (speculative.cpp
expects `ctx_dft` + a `dflash.block_size` metadata key); (3) measure baseline MTP/DFlash
acceptance rate — the core transferable number.


### 2026-09-07 (cont.) — P1.3b: **the headline result so far**

Ran matched Thai/English prompts through MANGO with and without the model's own MTP
drafter (`--spec-type draft-mtp`, llama.cpp b10068 built from the chimera vendor tree):

| spec type | lang | tok/s | mean acc len | **accept rate** |
|---|---|---|---|---|
| none | en | 17.99 | — | — |
| none | th | 18.00 | — | — |
| draft-mtp | en | 14.75 | 3.10 | **70.5%** |
| draft-mtp | th | 7.70 | 1.62 | **20.7%** |

1. **Acceptance gap is 3.41×** (70.5% en vs 20.7% th). Only 33% of Thai drafts landed any
   token at all, vs 85% for English. Thai discarded 1850 draft tokens vs English's 350.
   **This is distributional — it transfers to A100 unchanged.**
2. **Turning on MTP speculation made throughput worse on M4** — 18% slower in English,
   **2.34× slower in Thai**. The tok/s sign is M4-specific (81% of a 120 GB/s bandwidth
   ceiling) and must be re-measured on A100, but the acceptance numbers are final.
3. Baseline decode is language-neutral (17.99 vs 18.00) — as bandwidth-bound decode should
   be. So the entire gap comes from speculation behaviour, not from the language itself.

Detail + caveats: [`results/P13b_FINDINGS.md`](./results/P13b_FINDINGS.md).
Earlier the same day, P1.3a measured Thai at **1.845× more tokens** than English for the
same content ([`results/P13a_FINDINGS.md`](./results/P13a_FINDINGS.md)).

**This gives the project a real baseline to beat**: low, wasteful acceptance under a fixed
draft budget is exactly the regime `caddtree_budget.rs` adapts to — P2.3 now has a measured
target rather than a hypothesis.

**Next:** P1.4 energy harness; add `draft-dflash` (drafter already downloaded) and
`ngram-*` to the matrix; repeat runs for error bars.


### 2026-09-07 (cont.) — P1.3c + P1.4: **Phase 1 complete, and speculation now wins**

**P1.3c budget sweep corrects P1.3b's headline.** Sweeping `--spec-draft-n-max` over both
languages (baselines: en 17.99 / th 18.00):

| n_max | en tok/s (vs base) | th tok/s (vs base) |
|---|---|---|
| **1** | **21.49 (+19.5%)** | 16.05 (−10.8%) |
| 2 | 19.31 (+7.3%) | 11.70 (−35.0%) |
| 3 *(default)* | 14.76 (−18.0%) | 7.70 (−57.2%) |
| 5 | 13.11 (−27.1%) | 5.71 (−68.3%) |
| 8 | 14.11 (−21.6%) | 5.73 (−68.2%) |

1. **Speculation does win** — at `n_max=1`, English hits 21.49 tok/s, **+19.5% over
   baseline and 96% of the M4's theoretical bandwidth ceiling**. P1.3b's "speculation is a
   net loss" was an artefact of llama.cpp's default `n_max=3`, not a property of the method.
2. **The optimum is language-dependent**: English wants n=1, Thai wants **n=0 (off)** — no
   Thai budget beats baseline. A static config cannot serve a bilingual model. **P2.3
   (CaDDTree adaptive budget) now has a measured target, not a hypothesis.**
3. Accept rate decays monotonically with budget (en 86.6→41.0%, th 39.2→9.0%) and waste
   grows superlinearly (th: 551 → 5,286 discarded draft tokens), compounding the GDN
   recurrent-state rollback tax (`n_rs_seq`=3 observed, `[EXPERIMENTAL]` in llama.h).

Detail + caveats: [`results/P13c_FINDINGS.md`](./results/P13c_FINDINGS.md).

**P1.4 energy harness built and math-verified** — `experiments/p14_energy/power.py`, one
interface with `nvml` / `powermetrics` / `none` backends so the same script runs on M4 and
A100 unchanged. 100 ms sampling, idle-baseline subtraction, trapezoidal integration, ±5%
proportional error band, never TDP-derived. Unit-tested against analytic cases (constant
power → 2000.0 J expected/got; ramp → 1000.0 J expected/got).
**Untested gap: the `nvml` backend has never run against real hardware** — exercise it in
the first cheap A100 session. Detail: [`results/P14_FINDINGS.md`](./results/P14_FINDINGS.md).

**Phase 1 is complete.** Delivered without any integration work: a bilingual token-density
result, a bilingual acceptance-gap result, a budget-sweep result that turns speculation from
a loss into a +19.5% win, and a working A100-ready energy instrument.

**Next options:** repeat runs for error bars · widen the matrix (`draft-dflash` drafter is
already downloaded, plus `ngram-*`) · start Phase 2 (Rust bin + `extern "C"` shim → katgpt-rs).


### 2026-09-07 (cont.) — handoff prep for the GPU machine

Repo made portable and ready to push. No new measurements in this entry.

- **`.gitignore`** added: `models/` (6.4 GB), `**/target/` (191 MB of Rust build output),
  `results/*_logs/` (26 MB of verbosity-10 server traces). **What ships is 0.2 MB** — all
  findings docs plus the parsed JSON behind every number.
- **`scripts/fetch_models.py`** re-downloads the gitignored GGUFs (`--draft` for the DFlash
  drafter, `--awq` for the 9.1 GB deployment build).
- **`scripts/build_llama.sh`** builds llama.cpp with CUDA/Metal auto-detected. Needed because
  packaged b9960 lacks the `--spec-type` selector; Phase 1 used a b10068 source build.
- **`experiments/common_env.py`** removes the hardcoded `/tmp/lcpp-build` path — binaries now
  resolve via `LLAMA_BIN_DIR` → repo-local build → the old dev path → `PATH`; `NGL` tunes
  offload for small-VRAM cards. Both sweep scripts patched to use it; all scripts compile.
- **Docs brought in line with reality**: `AGENT_HANDOFF.md` **rewritten to v2.0** (v1 predated
  every measurement), `APPROACH.md` **rewritten to v2** (v1 archived — it still specified
  Candle, which D1 replaced with llama.cpp+Rust), `PLAN.md` gains **Phase 1.5 (re-establish on
  CUDA)**, `README.md` leads with the headline results and a run recipe.
- **Two traps recorded in `AGENT_HANDOFF.md` §0** for the next agent: this project depends on
  sibling repos (`katgpt-rs`, `chimera/vendor/llama.cpp`) **by relative path**, so cloning
  `mango-a100` alone breaks; and the owner has referred to the GPU box as both an RTX 4060 8 GB
  and an RTX 3060 — they differ in VRAM, bandwidth, architecture and FP8, so
  `nvidia-smi --query-gpu=name,memory.total` must be run and recorded before planning.
- **Knowledge base updated**: new topic
  `knowledge-base/topics/bilingual-speculative-decoding.md` carries the four Phase-1 local-facts
  so future tasks reuse them without re-running anything; indexed in `knowledge-base/INDEX.md`.

**Next agent starts at `AGENT_HANDOFF.md` §0 → Phase 1.5.**


### 2026-09-07 (cont.) — hardware ambiguity resolved

Owner confirmed the GPU box is an **RTX 4060 8 GB on Windows 11** (Ada `sm_89`, 272 GB/s), not
a 3060. The earlier entry above recorded the ambiguity; this is its resolution — docs updated,
and the practical consequences are now in `AGENT_HANDOFF.md` §0:

- Q4_K_M target (5.77 GB) fits in the ~7 GB usable budget at **≈6.9 GB @16K context**, but it
  is tight — verify with `nvidia-smi` under load.
- **The DFlash drafter does not fit alongside it** (≈8.0 GB total) → use `--spec-type draft-mtp`,
  which needs no second model. `draft-dflash` would require a smaller target quant.
- **The AWQ W4A16 deployment build (9.08 GB) does not fit at all** → validating it is an A100-only
  task, and a known gap to disclose in any writeup.
- KV cache is unusually small (~34 KB/token — only 8 of 32 layers cache K/V), so long context is
  limited by weights, not KV.
- **Ada has FP8, A100 does not** — the single most dangerous difference; never tune an FP8 path.
- Expected baseline here: decode ceiling 50.7 tok/s, so roughly **40–45 tok/s** at llama.cpp's
  usual 80–85% efficiency (vs 17.99 on the M4).
