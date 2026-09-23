#!/usr/bin/env python3
"""P1.3b — bilingual speculative-decoding acceptance on MANGO1.5-Qwen3.5-9B.

One llama-server per (spec_type, language) cell: send every prompt of that language,
then read the cumulative `spec ... statistics` trace lines the server emits
(common_speculative_print_stats, SPC_TRC) for the aggregate acceptance of that cell.

Aggregate-per-cell (rather than per-request) is deliberate: the counters are cumulative
over the server's lifetime, so a fresh server per cell gives clean attribution.
"""
import json, re, subprocess, sys, time, urllib.request, os, signal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_env import llama_bin, n_gpu_layers
BIN = llama_bin("llama-server")
MODEL = ROOT / "models/MANGO1.5-Qwen3.5-9B-Q4_K_M.gguf"
DRAFT = ROOT / "models/Qwen3.5-9B-DFlash-q8_0.gguf"
PROMPTS = json.loads((Path(__file__).parent / "prompts.json").read_text())["pairs"]
PORT = 8231
N_PREDICT = 128

STAT_RE = re.compile(
    r"statistics\s+(\S+):.*?#gen drafts\s*=\s*(\d+),\s*#acc drafts\s*=\s*(\d+),"
    r"\s*#gen tokens\s*=\s*(\d+),\s*#acc tokens\s*=\s*(\d+)"
    r"(?:.*?#mean acc len\s*=\s*([\d.]+))?")


def wait_ready(timeout=300):
    for _ in range(timeout):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read()
            return True
        except Exception:
            time.sleep(1)
    return False


def complete(prompt):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=json.dumps({"prompt": prompt, "n_predict": N_PREDICT, "temperature": 0,
                         "cache_prompt": False}).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=1800).read())


def run_cell(spec_type, lang, logdir):
    log = logdir / f"server_{spec_type}_{lang}.log"
    cmd = [BIN, "-m", str(MODEL), "--port", str(PORT), "-ngl", n_gpu_layers(),
           "-c", "4096", "--verbosity", "10", "--no-warmup"]
    if spec_type != "none":
        cmd += ["--spec-type", spec_type]
    if spec_type == "draft-dflash":
        cmd += ["-md", str(DRAFT)]
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, preexec_fn=os.setsid)
    try:
        if not wait_ready():
            return {"spec_type": spec_type, "lang": lang, "error": "server did not become ready"}
        tok, ms = 0, 0.0
        for p in PROMPTS:
            r = complete(p[lang])
            t = r.get("timings", {})
            tok += t.get("predicted_n", 0)
            ms += t.get("predicted_ms", 0.0)
        time.sleep(2)
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=60)

    text = log.read_text(errors="ignore")
    stats = {}
    for m in STAT_RE.finditer(text):
        impl, gd, ad, gt, at, mean = m.groups()
        stats[impl] = {"gen_drafts": int(gd), "acc_drafts": int(ad),
                       "gen_tokens": int(gt), "acc_tokens": int(at),
                       "mean_acc_len": float(mean) if mean else None,
                       "token_accept_rate": int(at) / int(gt) if int(gt) else None}
    return {"spec_type": spec_type, "lang": lang, "predicted_tokens": tok,
            "predicted_ms": round(ms, 1),
            "tok_per_s": round(tok / (ms / 1000), 2) if ms else None,
            "spec_stats": stats}


def main():
    types = sys.argv[1:] or ["none", "draft-mtp"]
    logdir = ROOT / "results" / "p13b_logs"; logdir.mkdir(parents=True, exist_ok=True)
    out = []
    for st in types:
        for lang in ("en", "th"):
            print(f"--- running {st} / {lang} ...", flush=True)
            r = run_cell(st, lang, logdir)
            print("   ", json.dumps(r, ensure_ascii=False)[:400], flush=True)
            out.append(r)
            (ROOT / "results" / "p13b_acceptance.json").write_text(
                json.dumps(out, indent=2, ensure_ascii=False))
    print("\nwrote results/p13b_acceptance.json")


if __name__ == "__main__":
    main()
