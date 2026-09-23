//! E03 — Weaver retarget probe.
//!
//! Question: is `katgpt-speculative`'s Weaver corrector architecture-general
//! enough to retarget from Gemma2-2B (hidden 2304, the shape its real-checkpoint
//! test fixture hardcodes) to MANGO1.5-Qwen3.5-9B (hidden 4096, 16 heads)?
//!
//! This runs the SAME code path at both shapes with zero-initialized weights
//! (`WeaverWeights::zeros` — documented as a safe no-op producing zero residuals)
//! and checks the structural invariants hold at each: output shapes, finiteness,
//! and that corrected probabilities form a distribution over the K candidates.
//!
//! Zero weights mean this proves *shape/'plumbing' generality*, not numerical
//! quality — a trained MANGO-shaped checkpoint is still required for real
//! acceptance-rate numbers. That is E04, not this.

use katgpt_speculative::weaver::{WeaverConfig, WeaverCorrector, WeaverInput, WeaverWeights};
use std::time::Instant;

struct Shape {
    name: &'static str,
    hidden: usize,
    n_heads: usize,
    d_ff: usize,
    depth: usize,
    k: usize,
    vocab: usize,
}

fn probe(s: &Shape) -> bool {
    let cfg = WeaverConfig {
        hidden_dim: s.hidden,
        n_heads: s.n_heads,
        k_candidates: s.k,
        n_layer: 1,
        d_ff: s.d_ff,
        rms_eps: 1e-6,
        max_depth: s.depth,
    };
    let head_dim = cfg.head_dim();

    let t_alloc = Instant::now();
    let corrector = WeaverCorrector::from_weights(WeaverWeights::zeros(cfg));
    let alloc_ms = t_alloc.elapsed().as_secs_f64() * 1e3;

    // Non-degenerate synthetic inputs, sized from the shape under test.
    let h_verifier: Vec<f32> = (0..s.hidden).map(|i| 0.5 + 1e-4 * i as f32).collect();
    let h_dflash_owned: Vec<Vec<f32>> = (0..s.depth)
        .map(|d| (0..s.hidden).map(|i| 0.3 + 1e-4 * ((d * s.hidden + i) as f32)).collect())
        .collect();
    let topk_owned: Vec<Vec<u32>> = (0..s.depth)
        .map(|d| (0..s.k).map(|j| ((d * s.k + j) % s.vocab) as u32).collect())
        .collect();
    let logits_owned: Vec<Vec<f32>> = (0..s.depth)
        .map(|d| (0..s.k).map(|j| ((j as f32) * 0.01) - (d as f32) * 0.05).collect())
        .collect();
    let embedding: Vec<f32> = (0..s.vocab * s.hidden)
        .map(|i| ((i % 97) as f32) * 0.001)
        .collect();

    let h_dflash: Vec<&[f32]> = h_dflash_owned.iter().map(|v| v.as_slice()).collect();
    let topk_ids: Vec<&[u32]> = topk_owned.iter().map(|v| v.as_slice()).collect();
    let dflash_logits: Vec<&[f32]> = logits_owned.iter().map(|v| v.as_slice()).collect();

    let input = WeaverInput {
        h_verifier: &h_verifier,
        h_dflash: &h_dflash,
        topk_ids: &topk_ids,
        dflash_logits: &dflash_logits,
        embedding: &embedding,
        vocab_size: s.vocab,
    };

    let t = Instant::now();
    let out = corrector.correct(&input);
    let fwd_ms = t.elapsed().as_secs_f64() * 1e3;

    // ── invariants ───────────────────────────────────────────────────────
    let mut ok = true;
    let mut fail = |msg: String| { println!("    FAIL: {msg}"); ok = false; };

    if out.depth != s.depth { fail(format!("depth {} != {}", out.depth, s.depth)); }
    if out.k != s.k { fail(format!("k {} != {}", out.k, s.k)); }
    if out.corrected_probs.len() != s.depth {
        fail(format!("corrected_probs rows {} != {}", out.corrected_probs.len(), s.depth));
    }
    for (d, row) in out.corrected_probs.iter().enumerate() {
        if row.len() != s.k { fail(format!("depth {d}: probs len {} != {}", row.len(), s.k)); continue; }
        let sum: f32 = row.iter().sum();
        if !(sum.is_finite() && (sum - 1.0).abs() < 1e-3) {
            fail(format!("depth {d}: probs sum {sum} != 1.0"));
        }
        if row.iter().any(|p| !p.is_finite() || *p < 0.0) {
            fail(format!("depth {d}: non-finite or negative probability"));
        }
    }
    for (d, row) in out.corrected_logits.iter().enumerate() {
        if row.iter().any(|x| !x.is_finite()) { fail(format!("depth {d}: non-finite corrected logit")); }
    }
    // zero weights => zero residual (the documented no-op contract)
    let max_res = out.weaver_residual.iter().flatten().fold(0f32, |m, v| m.max(v.abs()));
    if max_res > 1e-6 { fail(format!("zero weights produced residual {max_res:.3e} (expected ~0)")); }

    println!(
        "  {:<12} hidden={:<5} heads={:<3} head_dim={:<4} D={} K={} V={:<6} | alloc {:>7.1} ms | correct {:>6.2} ms | {}",
        s.name, s.hidden, s.n_heads, head_dim, s.depth, s.k, s.vocab, alloc_ms, fwd_ms,
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

fn main() {
    println!("E03 — Weaver retarget probe (zero weights; shape/plumbing generality only)\n");
    let shapes = [
        // control: the shape the existing real-checkpoint fixture uses
        Shape { name: "Gemma2-2B", hidden: 2304, n_heads: 16, d_ff: 5824, depth: 4, k: 32, vocab: 911 },
        // paper default, second control
        Shape { name: "paper-2048", hidden: 2048, n_heads: 16, d_ff: 5824, depth: 8, k: 32, vocab: 1024 },
        // the target: MANGO1.5-Qwen3.5-9B text config
        Shape { name: "MANGO-9B", hidden: 4096, n_heads: 16, d_ff: 12288, depth: 4, k: 32, vocab: 1024 },
        // target at the model's real MTP-ish depth/K sweep, bigger K
        Shape { name: "MANGO-K128", hidden: 4096, n_heads: 16, d_ff: 12288, depth: 8, k: 128, vocab: 4096 },
    ];
    let mut all = true;
    for s in &shapes { all &= probe(s); }
    println!("\n{}", if all { "ALL SHAPES PASS — Weaver is shape-general; retarget to hidden=4096 is viable." }
                    else { "SOME SHAPES FAILED — see FAIL lines above." });
    std::process::exit(if all { 0 } else { 1 });
}
