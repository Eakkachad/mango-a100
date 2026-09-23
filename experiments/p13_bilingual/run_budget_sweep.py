#!/usr/bin/env python3
"""P1.3c — draft-budget sweep per language.

P1.3b showed MTP drafting at the default budget (--spec-draft-n-max 3) is a net LOSS on
this hardware, catastrophically so in Thai (20.7% acceptance, 2.34x slowdown). If the
optimal fixed budget differs by language, that is direct evidence for an adaptive budget
selector (CaDDTree, P2.3) — obtainable without any integration work.

Sweeps n_max over both languages and records throughput + acceptance for each cell.
"""
import json, os, re, signal, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_env import llama_bin, n_gpu_layers
BIN = llama_bin("llama-server")
MODEL = ROOT / "models/MANGO1.5-Qwen3.5-9B-Q4_K_M.gguf"
PROMPTS = json.loads((Path(__file__).parent / "prompts.json").read_text())["pairs"]
PORT, N_PREDICT = 8232, 128

STAT_RE = re.compile(
    r"statistics\s+(\S+):.*?#gen drafts\s*=\s*(\d+),\s*#acc drafts\s*=\s*(\d+),"
    r"\s*#gen tokens\s*=\s*(\d+),\s*#acc tokens\s*=\s*(\d+)"
    r"(?:.*?#mean acc len\s*=\s*([\d.]+))?")


def wait_ready(t=300):
    for _ in range(t):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read(); return True
        except Exception: time.sleep(1)
    return False


def cell(n_max, lang, logdir):
    log = logdir / f"sweep_n{n_max}_{lang}.log"
    cmd = [BIN, "-m", str(MODEL), "--port", str(PORT), "-ngl", n_gpu_layers(), "-c", "4096",
           "--verbosity", "10", "--no-warmup",
           "--spec-type", "draft-mtp", "--spec-draft-n-max", str(n_max)]
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, preexec_fn=os.setsid)
    try:
        if not wait_ready():
            return {"n_max": n_max, "lang": lang, "error": "server not ready"}
        tok = ms = 0
        t0 = time.time()
        for p in PROMPTS:
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/completion",
                data=json.dumps({"prompt": p[lang], "n_predict": N_PREDICT,
                                 "temperature": 0, "cache_prompt": False}).encode(),
                headers={"Content-Type": "application/json"})
            t = json.loads(urllib.request.urlopen(req, timeout=1800).read()).get("timings", {})
            tok += t.get("predicted_n", 0); ms += t.get("predicted_ms", 0.0)
        wall = time.time() - t0
        time.sleep(2)
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM); proc.wait(timeout=60)

    st = {}
    for m in STAT_RE.finditer(log.read_text(errors="ignore")):
        impl, gd, ad, gt, at, mean = m.groups()
        st[impl] = {"gen_drafts": int(gd), "acc_drafts": int(ad), "gen_tokens": int(gt),
                    "acc_tokens": int(at), "mean_acc_len": float(mean) if mean else None,
                    "token_accept_rate": int(at)/int(gt) if int(gt) else None,
                    "draft_tokens_wasted": int(gt) - int(at)}
    return {"n_max": n_max, "lang": lang, "predicted_tokens": tok,
            "predicted_ms": round(ms, 1), "wall_s": round(wall, 1),
            "tok_per_s": round(tok/(ms/1000), 2) if ms else None, "spec_stats": st}


def main():
    budgets = [int(x) for x in (sys.argv[1:] or [1, 2, 3, 5, 8])]
    logdir = ROOT / "results" / "p13c_logs"; logdir.mkdir(parents=True, exist_ok=True)
    out, dest = [], ROOT / "results" / "p13c_budget_sweep.json"
    for n in budgets:
        for lang in ("en", "th"):
            print(f"--- n_max={n} / {lang}", flush=True)
            r = cell(n, lang, logdir); out.append(r)
            s = (r.get("spec_stats") or {}).get("draft-mtp", {})
            print(f"    tok/s={r.get('tok_per_s')} acc_rate={s.get('token_accept_rate')} "
                  f"acc_len={s.get('mean_acc_len')} wasted={s.get('draft_tokens_wasted')}", flush=True)
            dest.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
