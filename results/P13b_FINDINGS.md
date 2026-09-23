# P1.3b — MTP speculative decoding collapses on Thai: 20.7% vs 70.5% acceptance

**Date:** 2026-09-07 · **Hardware:** Apple M4 base, 16 GB (120 GB/s)
**Model:** MANGO1.5-Qwen3.5-9B Q4_K_M · **Runtime:** llama.cpp b10068 (built from
`neural-engines/chimera/vendor/llama.cpp`), Metal · **Drafter:** the model's own MTP head
(`--spec-type draft-mtp`, no separate draft model)
**Repro:** `python3 experiments/p13_bilingual/run_acceptance.py none draft-mtp`
**Data:** `results/p13b_acceptance.json`, server traces in `results/p13b_logs/`

10 matched Thai/English prompt pairs, `n_predict=128`, `temperature=0`, `cache_prompt=false`,
one fresh server per (spec_type, language) cell.

## Result

| spec type | lang | tokens | tok/s | mean accepted length | **token accept rate** | drafts gen / accepted |
|---|---|---|---|---|---|---|
| none | en | 1247 | **17.99** | — | — | — |
| none | th | 1280 | **18.00** | — | — | — |
| draft-mtp | en | 1247 | **14.75** | 3.10 | **70.5%** | 399 / 340 |
| draft-mtp | th | 1280 | **7.70** | 1.62 | **20.7%** | 781 / 255 |

### 1. The bilingual acceptance gap is 3.4×

English MTP drafts are accepted at **70.5%**; Thai at **20.7%**. Mean accepted draft length
falls from 3.10 tokens to 1.62. Only 33% of Thai drafts had *any* token accepted (255/781)
versus 85% for English (340/399).

**Acceptance rate is a property of the model's own probability distributions, not of the
hardware** — this number transfers to A100 unchanged. It is the core transferable result
of the whole project so far.

### 2. Enabling MTP speculation made throughput *worse* in both languages — catastrophically in Thai

Baseline decode is language-neutral (17.99 en / 18.00 th — as expected, since decode is
bandwidth-bound and streams the same weights regardless of language). Turning on MTP
drafting **cost 18% throughput in English and 57% in Thai** (18.00 → 7.70 tok/s, a 2.3×
slowdown). The Thai drafter burned 2332 draft tokens to land 482 accepted, versus 1186 → 836
for English: roughly twice the wasted draft compute for half the yield.

**Direct hackathon consequence: naively switching on MTP speculative decoding for a Thai
workload on this stack would make the submission ~2.3× slower.** Any "speculative decoding
gives 2–3×" claim from the literature (all of it measured on English Llama-family models —
see `knowledge-base/topics/speculative-decoding-and-pruning.md`) does not survive contact
with this model on Thai.

## What this does and does not prove

**Transfers to A100 (distributional, hardware-independent):** the 70.5% / 20.7% acceptance
split, the 3.10 / 1.62 accepted-length split, the draft-waste ratio.

**Does NOT transfer (M4-specific):** the tok/s figures and therefore the *sign* of the
throughput effect. The M4 base runs decode at 81% of its 120 GB/s bandwidth ceiling
(96.6 GB/s effective, 22.3 tok/s theoretical for a 5.37 GiB model); an A100's 2039 GB/s
puts it in a completely different compute-to-bandwidth regime where the same acceptance
rate may still yield a net win. **Re-measure the throughput effect on A100 before drawing
any conclusion about whether speculation pays there.** The acceptance numbers, however, are
already final.

## Why this is the project's opening

The failure mode is precisely what the katgpt-rs stack was built to attack:

- A fixed draft budget wastes compute when acceptance is low — which is exactly what
  `caddtree_budget.rs` (CaDDTree, arXiv:2606.01813 + BASTION, arXiv:2605.29727) adapts.
  **A budget selector that shrinks the draft on Thai and grows it on English is a directly
  measurable win against this baseline, and needs no training.** That is P2.3, and this run
  is its baseline.
- `weaver.rs` corrects draft logits before verification — the low-acceptance regime is
  where a corrector has the most headroom (though it needs a trained checkpoint; P2.2).

## Caveats

n = 10 prompt pairs per cell, single run, no repeats — no error bars yet. Run ≥3 repeats and
widen the prompt set before publishing. Author-written translations, matched by meaning, not
a standard parallel corpus. Thai outputs are longer in tokens for the same content (1.85×,
see `P13a_FINDINGS.md`), so per-cell token totals are not directly comparable — the accept
*rates* are ratios and unaffected.
