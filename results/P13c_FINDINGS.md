# P1.3c — The optimal draft budget is language-dependent, and llama.cpp's default is wrong for this model

**Date:** 2026-09-07 · **Hardware:** Apple M4 base, 16 GB (120 GB/s)
**Model:** MANGO1.5-Qwen3.5-9B Q4_K_M · **Runtime:** llama.cpp b10068 (self-built), Metal
**Drafter:** the model's own MTP head (`--spec-type draft-mtp`)
**Repro:** `python3 experiments/p13_bilingual/run_budget_sweep.py 1 2 3 5 8`
**Data:** `results/p13c_budget_sweep.json`, traces in `results/p13c_logs/`

10 matched prompt pairs, `n_predict=128`, `temperature=0`, `cache_prompt=false`,
fresh server per cell. Baselines from P1.3b: **en 17.99 / th 18.00 tok/s** with speculation off.

| n_max | lang | tok/s | vs. baseline | accept rate | mean acc len | draft tokens wasted |
|---:|---|---:|---:|---:|---:|---:|
| **1** | en | **21.49** | **+19.5%** | 86.6% | 1.87 | 88 |
| 2 | en | 19.31 | +7.3% | 77.6% | 2.55 | 216 |
| 3 *(llama.cpp default)* | en | 14.76 | −18.0% | 70.5% | 3.10 | 350 |
| 5 | en | 13.11 | −27.1% | 57.8% | 3.84 | 666 |
| 8 | en | 14.11 | −21.6% | 41.0% | 4.18 | 1350 |
| 1 | th | 16.05 | −10.8% | 39.2% | 1.39 | 551 |
| 2 | th | 11.70 | −35.0% | 27.1% | 1.54 | 1194 |
| 3 *(llama.cpp default)* | th | 7.70 | −57.2% | 20.7% | 1.62 | 1850 |
| 5 | th | 5.71 | −68.3% | 13.6% | 1.67 | 3225 |
| 8 | th | 5.73 | −68.2% | 9.0% | 1.70 | 5286 |

## Three results

### 1. Speculation does win — at a budget nobody ships by default

At `n_max=1`, English reaches **21.49 tok/s, +19.5% over the no-speculation baseline** —
and **96% of the 22.3 tok/s theoretical ceiling** implied by the M4's 120 GB/s bandwidth
against a 5.37 GiB model. That is speculative decoding doing exactly what it is supposed to
do: breaking the memory-bandwidth wall by verifying more than one token per weight-streaming
pass.

llama.cpp's default of `n_max=3` **loses 18%** in English and **57%** in Thai. The earlier
P1.3b conclusion ("MTP speculation is a net loss") was an artefact of that default, not a
property of the technique. Correcting the record: the technique works; the shipped default
does not fit this model.

### 2. The optimum differs by language — a static configuration cannot serve both

- **English: optimum at n_max = 1** (+19.5%).
- **Thai: no budget wins.** The best Thai cell (n_max=1) is still 10.8% *below* baseline, so
  the optimal policy for Thai on this hardware is **n_max = 0 — speculation off entirely.**

One static config cannot be right for both languages of a bilingual model. **This is the
direct, measured case for an adaptive budget selector** (`caddtree_budget.rs`, CaDDTree
arXiv:2606.01813 + BASTION arXiv:2605.29727) — P2.3 now has a quantified target instead of a
hypothesis, and the selector's job is concrete: pick 1 for English, 0 for Thai, per request.

### 3. Longer drafts decay acceptance and compound the rollback tax

Accept rate falls monotonically with budget in both languages (en 86.6 → 77.6 → 70.5 → 57.8
→ 41.0%; th 39.2 → 27.1 → 20.7 → 13.6 → 9.0%): later draft positions are harder to predict,
dragging the mean down. Waste grows superlinearly — Thai discards **5,286 draft tokens** at
n_max=8 versus 551 at n_max=1.

This compounds with the recurrent-state rollback tax specific to this architecture: 24 of
MANGO's 32 layers are Gated DeltaNet, and llama.cpp rolls their state back through bounded
per-token snapshots (`n_rs_seq`, observed = 3 at runtime, marked `[EXPERIMENTAL]` in
`llama.h`). Every rejected draft pays that cost, which is why the *same* accept-rate drop
hurts far more here than it would on a pure-attention model.

## Scope and caveats

- **Throughput figures are M4-specific.** The M4 runs at 81% of its bandwidth ceiling with
  speculation off; an A100's 2039 GB/s is a different compute-to-bandwidth regime, and the
  optimal budget there will very likely be **larger** (more compute to spend per byte
  streamed). Re-run this sweep on A100 before setting any production budget — the *shape* of
  the finding (language-dependent optimum) is what transfers, not the specific n_max values.
- **Accept rates are distributional and do transfer** — they are properties of the model's
  probability distributions, not of the hardware.
- n = 10 prompt pairs per cell, single run, no repeats: **no error bars yet.** The trends are
  monotonic and large (19.5% vs −57%), so the direction is safe; the exact percentages are
  not yet publication-grade. Run ≥3 repeats before quoting them in a paper.
- Author-written translations matched by meaning, not a standard parallel corpus.
- The n_max=8 cells break monotonicity slightly (en 14.11 > 13.11 at n=5; th 5.73 ≈ 5.71) —
  within run-to-run noise at n=1, another reason repeats are needed.
