#!/usr/bin/env python3
"""E01 — Target architecture probe.

THE gating question for this whole project: does MANGO1.5-Qwen3.5-9B contain
Gated DeltaNet (GDN) / linear-attention layers?

  - If YES  -> katgpt-rs's `gdn_tree_verify` (rollback-free GDN tree verification,
               arXiv:2607.06763) applies DIRECTLY to the target model. That is the
               strongest novelty asset we have.
  - If NO   -> that code is unusable for MANGO; weight shifts to Weaver + CaDDTree.

Reads config.json only (a few KB) — no weights downloaded.
Writes a machine-readable verdict to results/e01_arch_probe.json.
"""
import json
import sys
from pathlib import Path

CANDIDATES = [
    # the actual hackathon target, under whatever id it turns out to be published as
    "MANGO1.5-Qwen3.5-9B",
    "cmkl/MANGO1.5-Qwen3.5-9B",
    "CMKL-AI/MANGO1.5-Qwen3.5-9B",
    # base-architecture references
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-8B",
    # known GDN-bearing reference point, to prove the detector actually fires
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    # non-GDN control
    "Qwen/Qwen2.5-1.5B",
]

# substrings that indicate linear-attention / gated-deltanet style layers
GDN_MARKERS = (
    "linear_attn", "linear_attention", "deltanet", "delta_net", "gated_delta",
    "gdn", "mamba", "ssm", "conv1d", "recurrent",
)


def probe(model_id: str) -> dict:
    from transformers import AutoConfig
    out = {"model_id": model_id}
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    except Exception as e:  # gated repo, offline, or nonexistent
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return out
    d = cfg.to_dict()
    out["ok"] = True
    for k in ("model_type", "architectures", "num_hidden_layers", "hidden_size",
              "num_attention_heads", "num_key_value_heads", "intermediate_size",
              "vocab_size", "max_position_embeddings", "torch_dtype"):
        if k in d:
            out[k] = d[k]
    # layer_types / layer schedule is how hybrid models expose their mix
    for k in ("layer_types", "layers_block_type", "full_attention_interval",
              "linear_attn_config", "linear_conv_kernel_dim", "decoder_sparse_step"):
        if k in d:
            out[k] = d[k]
    hits = sorted({m for m in GDN_MARKERS
                   for k, v in d.items()
                   if m in str(k).lower() or m in str(v).lower()[:2000]})
    out["gdn_markers_hit"] = hits
    out["gdn_verdict"] = "LIKELY_HAS_GDN" if hits else "NO_GDN_MARKERS"
    return out


def main():
    ids = sys.argv[1:] or CANDIDATES
    results = [probe(m) for m in ids]
    for r in results:
        if r.get("ok"):
            print(f"[ok]   {r['model_id']}: {r.get('model_type')} "
                  f"L={r.get('num_hidden_layers')} h={r.get('hidden_size')} "
                  f"heads={r.get('num_attention_heads')}/kv={r.get('num_key_value_heads')} "
                  f"-> {r['gdn_verdict']} {r['gdn_markers_hit'] or ''}")
        else:
            print(f"[FAIL] {r['model_id']}: {r.get('error')}")
    out = Path(__file__).resolve().parents[2] / "results" / "e01_arch_probe.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
