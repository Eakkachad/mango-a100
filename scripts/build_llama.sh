#!/usr/bin/env bash
# Build llama.cpp with the DFlash/MTP speculative path.
#
# WHY A SOURCE BUILD: the packaged b9960 (brew/winget) has the nextn/MTP C API but NOT the
# `--spec-type` selector with draft-mtp / draft-dflash. Every Phase-1 result came from a
# build of the vendored b10068 tree. See results/P1_FINDINGS.md.
#
#   ./scripts/build_llama.sh            # auto-detect backend
#   BACKEND=cuda ./scripts/build_llama.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${LLAMA_SRC:-$HERE/../chimera/vendor/llama.cpp}"
BUILD="${LLAMA_BUILD_DIR:-$HERE/vendor/llama.cpp/build}"

[ -d "$SRC" ] || { echo "llama.cpp source not found at $SRC — set LLAMA_SRC"; exit 1; }

if [ -z "${BACKEND:-}" ]; then
  if command -v nvcc >/dev/null 2>&1 || command -v nvidia-smi >/dev/null 2>&1; then BACKEND=cuda
  elif [ "$(uname -s)" = "Darwin" ]; then BACKEND=metal
  else BACKEND=cpu; fi
fi
echo "building llama.cpp from $SRC  backend=$BACKEND  ->  $BUILD"

case "$BACKEND" in
  cuda)  FLAGS="-DGGML_CUDA=ON" ;;
  metal) FLAGS="-DGGML_METAL=ON" ;;
  *)     FLAGS="" ;;
esac

cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF $FLAGS
cmake --build "$BUILD" --target llama-cli llama-server llama-bench -j "${JOBS:-8}"

echo
echo "built. add this to your environment:"
echo "  export LLAMA_BIN_DIR=$BUILD/bin"
echo
echo "verify the DFlash/MTP path is present (must list draft-mtp and draft-dflash):"
echo "  $BUILD/bin/llama-cli --help | grep -A2 -- --spec-type"
