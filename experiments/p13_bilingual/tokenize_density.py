#!/usr/bin/env python3
"""P1.3a — tokenization density: Thai vs English on MANGO's own tokenizer.

Why this matters before any acceptance-rate number exists: if Thai needs materially
more tokens to express the same content, then tokens/sec and tokens-per-joule are
not comparable across languages on this model, and a bilingual serving benchmark
that reports a single tok/s figure is measuring something language-dependent.
"""
import json, statistics as st
from pathlib import Path
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
pairs = json.loads((Path(__file__).parent / "prompts.json").read_text())["pairs"]
tok = Tokenizer.from_file(hf_hub_download("cmkl/MANGO1.5-Qwen3.5-9B", "tokenizer.json"))

rows, ratios = [], []
for p in pairs:
    r = {"id": p["id"], "cat": p["cat"]}
    for lang in ("th", "en"):
        ids = tok.encode(p[lang]).ids
        r[f"{lang}_chars"] = len(p[lang])
        r[f"{lang}_tokens"] = len(ids)
        r[f"{lang}_chars_per_token"] = len(p[lang]) / len(ids)
    r["token_ratio_th_over_en"] = r["th_tokens"] / r["en_tokens"]
    ratios.append(r["token_ratio_th_over_en"])
    rows.append(r)

print(f"{'id':<14}{'cat':<11}{'th_tok':>7}{'en_tok':>7}{'th/en':>8}{'th c/t':>8}{'en c/t':>8}")
for r in rows:
    print(f"{r['id']:<14}{r['cat']:<11}{r['th_tokens']:>7}{r['en_tokens']:>7}"
          f"{r['token_ratio_th_over_en']:>8.2f}{r['th_chars_per_token']:>8.2f}{r['en_chars_per_token']:>8.2f}")

mean_r = st.mean(ratios); sd = st.pstdev(ratios)
print(f"\nToken ratio TH/EN over {len(ratios)} matched pairs: mean {mean_r:.3f}  sd {sd:.3f}"
      f"  min {min(ratios):.2f}  max {max(ratios):.2f}")
print(f"Mean chars/token — TH {st.mean(r['th_chars_per_token'] for r in rows):.2f}"
      f" | EN {st.mean(r['en_chars_per_token'] for r in rows):.2f}")

out = ROOT / "results" / "p13a_token_density.json"
out.write_text(json.dumps({"rows": rows, "mean_ratio_th_over_en": mean_r, "sd": sd}, indent=2, ensure_ascii=False))
print("wrote", out)
