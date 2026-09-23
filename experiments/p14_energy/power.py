#!/usr/bin/env python3
"""P1.4 — energy harness. One interface, three backends.

Methodology follows the two energy papers that independently converge on it
(knowledge-base/topics/energy-measurement.md):
  - sample instantaneous power at a fixed 100 ms cadence,
  - measure an idle baseline first and subtract it, so the reported figure is the
    *net* energy attributable to the workload,
  - integrate power over the workload window to get Joules,
  - report tokens-per-joule with an error band, never a bare number:
    nvidia-smi's real error is ±5% proportional, not the ±5W NVIDIA claims
    (arXiv 2312.02741), and TDP-derived estimates can overstate draw by up to 4.1x
    (arXiv 2505.06371) — so never estimate from TDP.

Backends
  nvml          A100 / any NVIDIA GPU. Prefers pynvml, falls back to `nvidia-smi
                --query-gpu=power.draw`. THIS is the backend the hackathon numbers
                come from.
  powermetrics  Apple Silicon. Requires root (`sudo`); reports clearly if it cannot run.
  none          No energy data. Everything else (timing, token counts) still works, so
                the same script runs unchanged on the M4 dev box.

Usage
    from power import PowerSampler
    with PowerSampler.auto() as s:      # or PowerSampler.for_backend("nvml")
        ... run the workload ...
    print(s.report(tokens=1234))
"""
from __future__ import annotations
import json, shutil, subprocess, sys, threading, time
from dataclasses import dataclass, field

SAMPLE_INTERVAL_S = 0.100     # 100 ms — the cadence both source papers use
IDLE_BASELINE_S   = 10.0      # 10 s idle measurement before the workload
NVIDIA_SMI_REL_ERR = 0.05     # ±5% proportional (arXiv 2312.02741)


class _Backend:
    name = "none"
    available = False
    rel_err = 0.0
    def read_watts(self) -> float | None: return None


class NvmlBackend(_Backend):
    name = "nvml"
    rel_err = NVIDIA_SMI_REL_ERR
    def __init__(self):
        self._h = None
        self._smi = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.available = True
        except Exception:
            if shutil.which("nvidia-smi"):
                self._smi = True
                self.available = True
    def read_watts(self):
        if self._h is not None:
            try: return self._pynvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0
            except Exception: return None
        if self._smi:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
                return float(out[0])
            except Exception: return None
        return None


class PowerMetricsBackend(_Backend):
    """Apple Silicon. `powermetrics` requires root; we probe non-interactively."""
    name = "powermetrics"
    rel_err = 0.0  # Apple does not publish an accuracy figure — treat as unknown, not zero
    def __init__(self):
        self.available = False
        self.reason = ""
        if not shutil.which("powermetrics"):
            self.reason = "powermetrics not installed"; return
        p = subprocess.run(["sudo", "-n", "powermetrics", "--samplers", "cpu_power",
                            "-n", "1", "-i", "200"], capture_output=True, text=True)
        if p.returncode == 0:
            self.available = True
        else:
            self.reason = ("powermetrics needs root and no passwordless sudo is configured — "
                           "run this script under `sudo`, or use backend 'none' on this box")
    def read_watts(self):
        try:
            out = subprocess.run(["sudo", "-n", "powermetrics", "--samplers", "cpu_power,gpu_power",
                                  "-n", "1", "-i", "200"], capture_output=True, text=True, timeout=10).stdout
            total = 0.0
            for line in out.splitlines():
                if "Power:" in line and "mW" in line:
                    try: total += float(line.split(":")[1].strip().split()[0]) / 1000.0
                    except Exception: pass
            return total or None
        except Exception:
            return None


@dataclass
class PowerSampler:
    backend: _Backend
    interval_s: float = SAMPLE_INTERVAL_S
    idle_baseline_w: float | None = None
    samples: list[tuple[float, float]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thr: threading.Thread | None = None

    @classmethod
    def for_backend(cls, name: str) -> "PowerSampler":
        b = {"nvml": NvmlBackend, "powermetrics": PowerMetricsBackend, "none": _Backend}[name]()
        return cls(backend=b)

    @classmethod
    def auto(cls) -> "PowerSampler":
        for B in (NvmlBackend, PowerMetricsBackend):
            b = B()
            if b.available: return cls(backend=b)
        return cls(backend=_Backend())

    def measure_idle(self, seconds: float = IDLE_BASELINE_S) -> float | None:
        if not self.backend.available: return None
        vals, t_end = [], time.time() + seconds
        while time.time() < t_end:
            w = self.backend.read_watts()
            if w is not None: vals.append(w)
            time.sleep(self.interval_s)
        self.idle_baseline_w = sum(vals) / len(vals) if vals else None
        return self.idle_baseline_w

    def _loop(self):
        while not self._stop.is_set():
            w = self.backend.read_watts()
            if w is not None: self.samples.append((time.time(), w))
            time.sleep(self.interval_s)

    def __enter__(self):
        if self.backend.available:
            self._stop.clear(); self._thr = threading.Thread(target=self._loop, daemon=True); self._thr.start()
        return self

    def __exit__(self, *exc):
        if self._thr: self._stop.set(); self._thr.join(timeout=5)
        return False

    def energy_joules(self, net: bool = True) -> float | None:
        """Trapezoidal integration of the power trace; idle baseline subtracted when net."""
        if len(self.samples) < 2: return None
        base = (self.idle_baseline_w or 0.0) if net else 0.0
        j = 0.0
        for (t0, w0), (t1, w1) in zip(self.samples, self.samples[1:]):
            j += ((max(w0 - base, 0.0) + max(w1 - base, 0.0)) / 2.0) * (t1 - t0)
        return j

    def report(self, tokens: int) -> dict:
        j = self.energy_joules()
        watts = [w for _, w in self.samples]
        r = {"backend": self.backend.name, "available": self.backend.available,
             "samples": len(self.samples), "interval_s": self.interval_s,
             "idle_baseline_w": self.idle_baseline_w, "tokens": tokens}
        if not self.backend.available:
            r["note"] = getattr(self.backend, "reason", "no power backend available on this host")
            return r
        r.update({"mean_w": round(sum(watts)/len(watts), 2) if watts else None,
                  "peak_w": round(max(watts), 2) if watts else None,
                  "net_energy_j": round(j, 2) if j else None,
                  "tokens_per_joule": round(tokens / j, 4) if j else None})
        if j and self.backend.rel_err:
            lo, hi = tokens / (j * (1 + self.backend.rel_err)), tokens / (j * (1 - self.backend.rel_err))
            r["tokens_per_joule_band"] = [round(lo, 4), round(hi, 4)]
            r["band_basis"] = f"±{self.backend.rel_err:.0%} proportional instrument error"
        return r


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "auto"
    s = PowerSampler.auto() if name == "auto" else PowerSampler.for_backend(name)
    print(f"backend={s.backend.name} available={s.backend.available}")
    if not s.backend.available:
        print("reason:", getattr(s.backend, "reason", "n/a"))
    else:
        print("idle baseline:", s.measure_idle(3.0), "W")
        with s:
            time.sleep(2.0)
        print(json.dumps(s.report(tokens=100), indent=2))
