# P1.1 / P1.2 — runtime + API availability

**Date:** 2026-09-07 · **Repro:** `results/P1_1_2_probe.log`

## P1.2 — VERDICT: YES, the nextn/MTP API is exported (with one catch)

`/opt/homebrew/Cellar/llama.cpp/9960/lib/libllama.dylib` exports **20 nextn symbols**,
including all four entry points D1 depends on:

```
llama_set_embeddings_nextn(llama_context*, bool, bool)
llama_get_embeddings_nextn(llama_context*)
llama_get_embeddings_nextn_ith(llama_context*, int)
llama_set_nextn_layer_offset(llama_context*, int)
```

Also present and useful for Weaver's `h_verifier` / `h_dflash` inputs:
`llama_get_embeddings_layer_inp(llama_context*, uint32_t)` and
`llama_set_embeddings_layer_inp(...)`.

**⚠️ The catch: these are C++-mangled, not `extern "C"`.** The symbols read
`__Z26llama_set_embeddings_nextnP13llama_contextbb`, i.e. the `llama-ext.h` staging API has
C++ linkage. Rust cannot bind them directly by name. **D1 therefore needs a ~20-line
`extern "C"` shim** compiled next to the Rust binary (a small `.cpp` that re-exports the
four functions with C linkage). That is a minor build step, not an architectural problem —
**D1 stands**, with the shim added to its scope.

## P1.1 — PARTIAL: MTP state access yes, DFlash drafter no

The brew **b9960** CLI has draft-*model* speculative decoding
(`--spec-draft-hf`, `--spec-draft-n-max` (default 3), `--spec-draft-n-min`,
`--spec-draft-p-split`, plus draft-side threading/cache-type flags) but **no
`--spec-draft-type` selector and no `draft-dflash` option.**

The DFlash drafter (`COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH`, registered as `"draft-dflash"`)
exists in the llama.cpp vendored at `neural-engines/chimera/vendor/llama.cpp` at **b10068
(2026-07-18)** — i.e. DFlash landed *after* the b9960 the brew formula installs.

**Consequence:** the installed binary is enough to *read MTP hidden states* but not to run
DFlash drafting end to end. To get acceptance-rate numbers we must build llama.cpp from the
vendored b10068 source (or newer upstream).

## Follow-on questions this opens

1. Does a **DFlash draft model exist for MANGO/Qwen3.5**? `common/speculative.cpp` asserts
   DFlash "requires ctx_tgt and ctx_dft" and reads a `dflash.block_size` metadata key —
   implying a separate draft GGUF, not the in-model MTP head alone. If none is published,
   P1.3's acceptance measurement needs a different drafter (or we write the MTP self-draft
   loop ourselves via the exported nextn API — which is Phase 2 work brought forward).
2. Does upstream llama.cpp now expose an MTP *self*-speculation type (no separate draft
   model)? Worth re-checking upstream at build time.
