# MANGO-A100

**Porting `katgpt-rs`'s speculative-decoding and KV-quantization research onto CUDA, to optimize
MANGO1.5-Qwen3.5-9B inference throughput and energy efficiency on NVIDIA A100 — for Green Mind AI
Hackathon 2026 (CMKL), and as a continuing research track afterward.**

Status: **Phase 1 complete (2026-09-07), measured on an Apple M4.** Next up is Phase 1.5 —
re-establishing the same results on CUDA. Hackathon starts ~2026-10-01.

**Headline results so far** (all reproducible, raw data in `results/`):
- Thai costs **1.845× more tokens** than English for the same content — so tokens/sec and
  tokens-per-joule are not comparable across languages on this model.
- MTP draft acceptance is **70.5% in English but 20.7% in Thai** — a 3.41× gap.
- **The optimal draft budget is language-dependent.** At `n_max=1`, English reaches
  **+19.5% over baseline** (96% of the M4's bandwidth ceiling); no Thai budget beats baseline
  at all. llama.cpp's shipped default (`n_max=3`) *loses* 18% in English and 57% in Thai.

**If you are an agent picking this up on another machine, read
[AGENT_HANDOFF.md](./AGENT_HANDOFF.md) §0 first.**

---

## Read these first, in order

1. **[AGENT_HANDOFF.md](./AGENT_HANDOFF.md)** — mission, key decisions and *why*, hardware constraints,
   what to read before writing any code, immediate next actions.
2. **[RESEARCH_REFERENCES.md](./RESEARCH_REFERENCES.md)** — verified literature findings (confirmed vs.
   refuted claims), open questions, and the local `katgpt-rs` component map.
3. **[APPROACH.md](./APPROACH.md)** — technical strategy, engineering guidelines, and definition of done.
4. **[PLAN.md](./PLAN.md)** — the week-by-week phased plan up to the hackathon and beyond.
5. **[PROGRESS.md](./PROGRESS.md)** — living log. Update this every session; read it to see what's
   actually been done vs. just planned.
6. **`results/`** — every findings file plus the raw JSON behind it.

## Running it

Models are gitignored (6.4 GB). On a fresh machine:

```bash
python3 scripts/fetch_models.py     # target model (5.8 GB); --draft adds the DFlash drafter
./scripts/build_llama.sh            # CUDA/Metal auto-detected; prints LLAMA_BIN_DIR to export
python3 experiments/p13_bilingual/run_budget_sweep.py 1 2 3 5 8
python3 experiments/p14_energy/power.py auto
```

A packaged llama.cpp (brew/winget b9960) will **not** work for the speculative experiments —
it lacks the `--spec-type` selector. Build from source; see `results/P1_FINDINGS.md`.

## One-line orientation

- **Target model:** MANGO1.5-Qwen3.5-9B — Qwen3.5 architecture, Thai-English bilingual, 9.2B
  params, and **a hybrid: 24 of its 32 layers are Gated DeltaNet, 8 are full attention**, plus
  a built-in MTP (multi-token prediction) head. That architecture is why this project exists.
- **Target hardware:** NVIDIA A100 80GB (remote, queued, research-permissioned).
- **Dev hardware:** Apple M4 base 16GB (Phase 1, done) → **RTX 4060 8GB / Windows 11**
  (Phase 1.5 — see `AGENT_HANDOFF.md` §0 for its VRAM limits) → A100 for the final numbers.
- **Core bet:** don't abandon `katgpt-rs` for an off-the-shelf build. **Rust orchestrates,
  llama.cpp is the kernel library** (decision D1 in `PLAN.md`): llama.cpp runs the model — it
  is the only runtime supporting `qwen35` + MTP/DFlash today — while katgpt-rs supplies the
  research components: `caddtree_budget` (adaptive draft budget), `weaver` (draft-logit
  corrector), and `gdn_tree_verify` (rollback-free tree verification against GDN layers,
  implementing Oda et al. arXiv:2607.06763).
- **Why that bet, in one measured sentence:** llama.cpp rolls back GDN recurrent state through
  bounded snapshots after every rejected draft, which is exactly why Thai's 20.7% acceptance
  costs 57% throughput — and `gdn_tree_verify` removes that cost by construction.
- **Full research report (external literature):** [The MANGO Inference Playbook](https://claude.ai/code/artifact/6b913516-9c95-42bd-ae2f-b8ab7f2afea8) (published artifact).
