#!/usr/bin/env python3
"""Fetch raw config.json (bypasses AutoConfig's architecture registry, which does
not yet know model_type `qwen3_5`)."""
import json, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

IDS = sys.argv[1:] or ["cmkl/MANGO1.5-Qwen3.5-9B", "Qwen/Qwen3.5-9B"]
GDN = ("linear_attn","linear_attention","deltanet","delta_net","gated_delta","gdn",
       "mamba","ssm","conv1d","recurrent","layer_types","full_attention")

out = {}
for mid in IDS:
    try:
        p = hf_hub_download(mid, "config.json")
        cfg = json.loads(Path(p).read_text())
        out[mid] = cfg
        print(f"\n===== {mid} =====")
        print(json.dumps(cfg, indent=2, ensure_ascii=False)[:2600])
        hits = sorted({m for m in GDN for k, v in cfg.items()
                       if m in str(k).lower() or m in str(v).lower()[:3000]})
        print(f"\n>>> GDN/linear-attention markers: {hits or 'NONE'}")
    except Exception as e:
        print(f"[FAIL] {mid}: {type(e).__name__}: {str(e)[:300]}")
Path("results/e01_raw_configs.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("\nwrote results/e01_raw_configs.json")
