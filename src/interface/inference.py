"""The repair methods of the interface (checkpoints loaded on first use, then kept).

Methods (settings frozen in configs/app.yaml, chosen in the Day 7 selection):
  size_adaptive_hybrid  matched U-Net sketch -> noised to t0 -> diffusion (constraint fine-tune)
                        finishes it; t0 = 400 / 600 / 700 for small / medium / large holes
  hybrid_600            the same with t0 = 600 for every hole
  unet_matched          the matched U-Net alone (one forward pass)
  diffusion_refine600   pure diffusion: its own one-shot guess, refined from t0 = 600
  nearest_fill          baseline: every hole pixel copies the nearest known pixel
The diffusion finish is stochastic DDIM (eta 1) with the known-region reset
(src.sampling_experiment.refine), exactly as evaluated.

Checkpoints are loaded lazily: a network is read from disk the first time a
method needs it and then kept (nearest fill needs none, the U-Net only the
U-Net, refine-600 only the diffusion model, the hybrids both). There are only
two networks, so at most two models are ever in memory (~0.2 GB in fp32).

Hole size at inference: the mask area as % of the building, where the
building (wall + door + window + room) is measured on the U-Net-completed
plan — the clean plan is unknown here. On the val benchmark this gives the
true size bin in 7,771 of 7,772 cases.
"""

import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import yaml

from src.data.corruptions import BUILDING, UNKNOWN, to_model_input
from src.diagnose_diffusion import one_shot
from src.eval.baselines import nearest_fill
from src.evaluate import load_trained
from src.sampling_experiment import refine
from src.utils import project_path

METHOD_NAMES = {
    "size_adaptive_hybrid": "Size-adaptive hybrid (recommended)",
    "hybrid_600": "Hybrid, t₀ = 600",
    "unet_matched": "Matched U-Net",
    "diffusion_refine600": "Diffusion, refine-600",
    "nearest_fill": "Nearest fill (baseline)",
}
STOCHASTIC = {"size_adaptive_hybrid", "hybrid_600", "diffusion_refine600"}   # the seed matters
NEEDS = {                                   # the networks (checkpoint keys) each method uses
    "size_adaptive_hybrid": ("unet_matched", "diffusion"),
    "hybrid_600": ("unet_matched", "diffusion"),
    "unet_matched": ("unet_matched",),
    "diffusion_refine600": ("diffusion",),
    "nearest_fill": (),
}


class CheckpointMissing(RuntimeError):
    """A checkpoint named in configs/app.yaml does not exist."""


@dataclass
class Repair:
    labels: np.ndarray                 # repaired 128x128 label map (known pixels unchanged)
    method: str
    seconds: float
    details: dict = field(default_factory=dict)


def size_bin(mask, completed, edges):
    """'small' / 'medium' / 'large': mask area as % of the building in the completed plan."""
    building = np.isin(completed, BUILDING).sum()
    pct = 100.0 * mask.sum() / max(building, 1)
    return ("small" if pct < edges[0] else "medium" if pct < edges[1] else "large"), pct


def check_checkpoints(cfg):
    missing = {k: p for k, p in cfg["checkpoints"].items() if not project_path(p).exists()}
    if missing:
        lines = "\n".join(f"  - {k}: {project_path(p)}" for k, p in missing.items())
        raise CheckpointMissing(
            "Missing model checkpoint(s):\n" + lines + "\n"
            "Download them from the Kaggle kernel outputs (see docs/kaggle_workflow.md: "
            "floorplan-unet-matched and floorplan-finetune-constraints) or change the paths in configs/app.yaml.")


class Repairer:
    """Runs any method on one plan; each network is loaded on first use and then kept."""

    def __init__(self, cfg, device=None):
        check_checkpoints(cfg)
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}                           # checkpoint key -> loaded network (filled on first use)
        corruption = yaml.safe_load(open(project_path("configs/corruption.yaml"), encoding="utf-8"))
        self.edges = corruption["size_bins"]["edges"]

    def model(self, key):
        """The network for a checkpoint key ('unet_matched' or 'diffusion'), loaded on first use."""
        if key not in self.models:
            self.models[key], _ = load_trained(project_path(self.cfg["checkpoints"][key]), self.device)
        return self.models[key]

    @property
    def unet(self):
        return self.model("unet_matched")

    @property
    def diffusion(self):
        return self.model("diffusion")

    def condition(self, plan, mask):
        corrupted = plan.copy()
        corrupted[mask] = UNKNOWN                  # the hole is unknown to the model
        return torch.from_numpy(to_model_input(corrupted, mask)[None]).to(self.device)

    @torch.inference_mode()
    def repair(self, plan, mask, method, seed=0, progress=None):
        """plan: 128x128 label map (UNKNOWN allowed inside the mask); mask: bool 128x128.
        progress(done, total, text): optional callback for a progress bar."""
        if method not in METHOD_NAMES:
            raise ValueError(f"unknown method {method}")
        if not mask.any():
            raise ValueError("empty mask")
        s = self.cfg["sampling"]
        report = progress or (lambda *a: None)
        for key in NEEDS[method]:
            if key not in self.models:
                report(0, 1, f"loading the {key} checkpoint (first use only)")
                self.model(key)
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        t_start = time.perf_counter()
        details = {}
        if method == "nearest_fill":
            corrupted = plan.copy()
            corrupted[mask] = UNKNOWN
            labels = nearest_fill({"corrupted": corrupted[None], "masks": mask[None]})[0]
        else:
            cond = self.condition(plan, mask)
            if method == "diffusion_refine600":
                report(0, 1, "one-shot guess")
                start, t0 = one_shot(self.diffusion, cond, False, gen), s["refine_t0"]
            else:
                report(0, 1, "U-Net sketch")
                start = self.unet.predict(cond)
                if method == "unet_matched":
                    t0 = None
                else:
                    sketch = np.where(mask, start[0].cpu().numpy(), plan)
                    details["size_bin"], details["mask_pct_of_building"] = size_bin(mask, sketch, self.edges)
                    t0 = (s["size_adaptive_t0"][details["size_bin"]] if method == "size_adaptive_hybrid"
                          else s["hybrid_t0"])
            if t0 is not None:
                details["t0"] = t0
                labels = refine(self.diffusion, cond, start, t0, s["ddim_steps"], gen,
                                callback=lambda done, total: report(done, total, f"denoising step {done}/{total}"))
            else:
                labels = start
            labels = labels[0].cpu().numpy().astype(np.uint8)
        labels = np.where(mask, labels, plan).astype(np.uint8)     # known pixels never change
        report(1, 1, "done")
        return Repair(labels, method, time.perf_counter() - t_start, details)


def load_config():
    """configs/app.yaml, or the file named by the environment variable FLOORPLAN_APP_CONFIG."""
    path = os.environ.get("FLOORPLAN_APP_CONFIG") or project_path("configs/app.yaml")
    return yaml.safe_load(open(path, encoding="utf-8"))


def describe(repair):
    """One line about how the repair was made (for the interface)."""
    d = repair.details
    parts = [METHOD_NAMES[repair.method]]
    if "size_bin" in d:
        parts.append(f"hole = {d['mask_pct_of_building']:.1f}% of the building → {d['size_bin']}")
    if "t0" in d:
        parts.append(f"t₀ = {d['t0']}")
    return " · ".join(parts)
