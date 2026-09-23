"""Shared environment resolution — keeps the experiment scripts portable across the
M4 dev box, the Windows/RTX machine, and the A100.

Resolution order for the llama.cpp binaries:
  1. $LLAMA_BIN_DIR                      (explicit override — set this on a new machine)
  2. ./vendor/llama.cpp/build/bin        (repo-local build, what scripts/build_llama.sh makes)
  3. /tmp/lcpp-build/bin                 (the macOS dev build used for the Phase-1 results)
  4. whatever is on PATH                 (e.g. a brew/winget install)

Note the b9960 brew build does NOT have the DFlash/MTP `--spec-type` selector; the
Phase-1 numbers were produced with a build from the vendored b10068 source. See
results/P1_FINDINGS.md.
"""
import os, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXE = ".exe" if os.name == "nt" else ""


def llama_bin(name: str) -> str:
    """Absolute path to a llama.cpp binary, or the bare name if only PATH has it."""
    cands = []
    if os.environ.get("LLAMA_BIN_DIR"):
        cands.append(Path(os.environ["LLAMA_BIN_DIR"]))
    cands += [REPO / "vendor" / "llama.cpp" / "build" / "bin", Path("/tmp/lcpp-build/bin")]
    for d in cands:
        p = d / f"{name}{EXE}"
        if p.exists():
            return str(p)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"could not find '{name}'. Set LLAMA_BIN_DIR to your llama.cpp build/bin, "
        f"or run scripts/build_llama.sh. Tried: {[str(c) for c in cands]}")


def model_path(filename: str) -> Path:
    p = Path(os.environ.get("MANGO_MODEL_DIR", REPO / "models")) / filename
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run scripts/fetch_models.py first")
    return p


def n_gpu_layers() -> str:
    """-ngl value. 99 = offload everything (default); override on small-VRAM cards."""
    return os.environ.get("NGL", "99")
