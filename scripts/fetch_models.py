#!/usr/bin/env python3
"""Download the GGUF artifacts the experiments need (they are gitignored).

  python3 scripts/fetch_models.py            # target model only (5.8 GB) — enough for
                                             # everything Phase 1 measured, since
                                             # --spec-type draft-mtp uses the model's own
                                             # MTP head and needs no separate drafter
  python3 scripts/fetch_models.py --draft    # also the DFlash drafter (1.1 GB)
  python3 scripts/fetch_models.py --awq      # the CMKL AWQ W4A16 deployment artifact
                                             # (9.1 GB, CUDA only — will NOT fit an 8 GB card)
"""
import argparse, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

DEST = Path(__file__).resolve().parents[1] / "models"
TARGET = ("CMKL/MANGO1.5-Qwen3.5-9B-GGUF", "MANGO1.5-Qwen3.5-9B-Q4_K_M.gguf", "5.8 GB")
DRAFT  = ("lym00/Qwen3.5-9B-DFlash-GGUF-Test", "Qwen3.5-9B-DFlash-q8_0.gguf", "1.1 GB")


def get(repo, fn, size):
    print(f"→ {repo}/{fn}  ({size})")
    p = hf_hub_download(repo, fn, local_dir=str(DEST))
    print(f"  ok: {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true", help="also fetch the DFlash drafter")
    ap.add_argument("--awq", action="store_true", help="also fetch the AWQ W4A16 deployment build")
    a = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    get(*TARGET)
    if a.draft:
        get(*DRAFT)
    if a.awq:
        from huggingface_hub import snapshot_download
        print("→ CMKL/MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp  (9.1 GB, CUDA only)")
        print(snapshot_download("CMKL/MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp",
                                local_dir=str(DEST / "awq-w4a16-mtp")))
    print("\ndone.")


if __name__ == "__main__":
    main()
