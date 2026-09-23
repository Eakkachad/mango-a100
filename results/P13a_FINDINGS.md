# P1.3a — Thai costs 1.85× more tokens than English on MANGO's own tokenizer

**Date:** 2026-09-07 · **Repro:** `python3 experiments/p13_bilingual/tokenize_density.py`
**Data:** `results/p13a_token_density.json` · **Prompts:** `experiments/p13_bilingual/prompts.json`
(10 matched Thai/English pairs — same task, same semantic content, across factual /
creative / structured / reasoning / code)

| metric | Thai | English |
|---|---|---|
| mean chars per token | **1.99** | **4.46** |
| token ratio TH/EN (per matched pair) | **mean 1.845, sd 0.310** (min 1.37, max 2.38) | 1.0 by definition |

Per-pair ratios: factual 1.53 / 1.91 · creative 1.94 / 2.38 · structured 1.55 / 1.71 ·
reasoning 1.37 / 1.74 · code 2.27 / 2.04.

## Why this matters for a bilingual serving benchmark

Expressing the *same content* in Thai costs ~1.85× the tokens. So on this model:

- **tokens/sec is not comparable across languages.** A system reporting 17.8 tok/s is
  delivering roughly half as much *content* per second in Thai as in English.
- **tokens-per-joule inherits the same distortion** — the Green Mind objective is energy
  per unit of delivered value, and a per-token metric understates Thai energy cost by the
  same ~1.85×.
- Any single-number throughput or efficiency claim for a Thai-English bilingual model is
  therefore **language-mix dependent** and must state the mix it was measured under.

**Proposed reporting rule for this project:** publish tokens/sec and tokens-per-joule
*per language*, plus a content-normalised figure (e.g. per-1k-characters or per matched
task) so the two languages can be compared honestly. This is cheap to do and, as far as
the surveyed literature goes, nobody reports it —
`knowledge-base/topics/speculative-decoding-and-pruning.md` records that every EAGLE /
Medusa / SpecDec / Cascade number found is English Llama-family.

## Caveats

Ten prompt pairs is a small sample: the ratio's spread (1.37–2.38) is wide and clearly
task-dependent (code, where much of the output is ASCII either way, behaves differently
from prose). Treat 1.85 as an order-of-magnitude anchor, not a precise constant — widen
the prompt set before putting a number in a paper. Translations are the author's own and
matched by meaning, not by a standard parallel corpus.
