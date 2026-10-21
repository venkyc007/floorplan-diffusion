"""Time one repair per method on this machine (CPU unless a GPU is available).

Uses 3 gallery cases (the first small, medium and large hole), seed 0, after
one warm-up repair. Also times loading each checkpoint (done once, on the
first repair that needs it). Writes outputs/stage1/interface/cpu_timing.md.

    python -m src.stage1.interface.time_methods
"""

import platform
import time

import numpy as np
import torch

from src.stage1.interface.inference import METHOD_NAMES, Repairer, load_config
from src.utils import project_path


def main():
    cfg = load_config()
    rep = Repairer(cfg)
    g = np.load(project_path(cfg["gallery"]))
    cases = {b: int(np.flatnonzero(g["size_bins"] == b)[0]) for b in ("small", "medium", "large")}
    loads = {}
    for key in ("unet_matched", "diffusion"):                                    # first-use loading cost
        t = time.perf_counter()
        rep.model(key)
        loads[key] = time.perf_counter() - t
    rep.repair(g["inputs"][0], g["masks"][0], "unet_matched")                  # warm-up
    rows = []
    for method in cfg["methods"]:
        times, notes = [], []
        for b, i in cases.items():
            r = rep.repair(g["inputs"][i], g["masks"][i], method, seed=0)
            times.append(r.seconds)
            notes.append(f"t₀ {r.details['t0']}" if "t0" in r.details else "")
            print(f"{method:22s} {b:6s} {r.seconds:6.2f} s {notes[-1]}", flush=True)
        rows.append((METHOD_NAMES[method], times, notes))
    device = (torch.cuda.get_device_name(0) if rep.device.type == "cuda"
              else f"CPU: {platform.processor() or platform.machine()}, {torch.get_num_threads()} threads")
    lines = ["# Repair time per method", "", f"One repair per method and hole size (seed 0, after a warm-up), "
             f"on **{device}**, PyTorch {torch.__version__}. Diffusion cost grows with t₀: the steps at or below "
             "t₀ of the 50-step DDIM schedule are run (t₀ = 400 → 21 steps, 600 → 31, 700 → 36).", "",
             "| method | small hole | medium hole | large hole |", "|---|---|---|---|"]
    for name, times, notes in rows:
        cells = [f"{t:.2f} s" + (f" ({n})" if n else "") for t, n in zip(times, notes)]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Loading a checkpoint (once, on the first repair that needs it): "
              + ", ".join(f"{k} {v:.1f} s" for k, v in loads.items()) + "."]
    out = project_path("outputs/stage1/interface/cpu_timing.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
