"""The large-hole FILL models of the novelty work: the Stage 2 16-class U-Net / diffusion (models16.py) plus the
optional Fourier context blocks (ffc.py).

    model = build_large(mcfg)            # mcfg["ffc"] = {enabled: true, at: [down2, mid, up0]} adds the blocks
    warm_start(model, "b4 best.pt")      # B4 weights into the U-Net; the new blocks start as the identity
    model = load_large("best.pt")        # a trained novelty checkpoint (its config says whether it has the blocks)
"""

import torch

from src.novelty.ffc import SpectralAdapters
from src.stage2.complete.models16 import build16


def build_large(mcfg):
    model = build16(mcfg)
    f = mcfg.get("ffc") or {}
    if f.get("enabled"):
        model.spectral = SpectralAdapters(model.net, f.get("at", ["down2", "mid", "up0"]),
                                          groups=min(8, mcfg.get("norm_num_groups", 32)))
    return model


def warm_start(model, path):
    """Load a 16-class checkpoint without Fourier blocks (B4) into `model`; only the new blocks may be missing."""
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.startswith("spectral.") for k in missing), (missing[:5], unexpected[:5])
    return model


def load_large(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_large(ck["config"]["model"])
    model.load_state_dict(ck["model"])
    return model.eval().to(device)
