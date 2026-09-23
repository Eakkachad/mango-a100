# P1.4 — Energy harness (built, math-verified, A100-ready)

**Date:** 2026-09-07 · **Code:** `experiments/p14_energy/power.py`

## What it does

One `PowerSampler` interface, three interchangeable backends, so the *same measurement
script* runs on the M4 dev box and on the A100 without a rewrite — which was the design
requirement in `PLAN.md` P1.4.

| backend | host | status |
|---|---|---|
| `nvml` | any NVIDIA GPU | prefers `pynvml`, falls back to `nvidia-smi --query-gpu=power.draw`. **This is the backend the hackathon numbers come from.** Untested — no NVIDIA hardware here yet. |
| `powermetrics` | Apple Silicon | detected, **unavailable without root**: probed non-interactively with `sudo -n` and reports a clear reason rather than hanging on a password prompt. Works when the script itself is run under `sudo`. |
| `none` | anything | no energy data; timing and token counts still recorded, so experiments run unchanged on the M4. |

`PowerSampler.auto()` picks the first available backend; on this Mac mini it correctly
falls back to `none` (no NVML, no passwordless sudo, and `AppleSmartBattery` reports
`Amperage=0 / Voltage=0` — a Mac mini has no battery to derive draw from).

## Methodology (follows the knowledge-base sources)

- **100 ms sampling cadence** — the interval both energy papers in
  `knowledge-base/topics/energy-measurement.md` independently converge on
  (Samsi et al. HPEC 2023; arXiv 2512.01644).
- **Idle baseline measured first and subtracted**, so the reported figure is net energy
  attributable to the workload, not the machine's floor.
- **Trapezoidal integration** of the power trace over the workload window → Joules.
- **Never estimate from TDP** — arXiv 2505.06371 measured TDP-based estimation
  overstating real draw by up to 4.1×.
- **Always report an error band**: nvidia-smi's real error is ±5% *proportional*, not the
  ±5W NVIDIA claims (arXiv 2312.02741). `tokens_per_joule_band` carries this.

## Verification

The integration math is unit-tested against two analytic cases (`power.py` self-test):

| case | expected | got |
|---|---|---|
| constant 300 W for 10 s, 100 W idle baseline → net | 2000.0 J | **2000.0 J** |
| tokens-per-joule at 4000 tokens | 2.0 | **2.0** |
| ±5% band | ~[1.905, 2.105] | **[1.9048, 2.1053]** |
| linear ramp 100→300 W over 10 s, 100 W baseline (triangle) | 1000.0 J | **1000.0 J** |

## Known gaps

- The `nvml` backend has **not been executed against real hardware** — it is written but
  unproven. Exercise it on the first (cheap) A100 session before relying on it, per
  `PLAN.md` P3.3.
- `powermetrics` parsing sums any `Power: … mW` lines from the `cpu_power,gpu_power`
  samplers; the field set should be checked against a real root-run before its numbers are
  quoted. Apple publishes no accuracy figure, so `rel_err` is left at 0 and the band is
  omitted for that backend rather than fabricated.
- No energy number has been measured yet on any host — this deliverable is the
  *instrument*, not a result.
