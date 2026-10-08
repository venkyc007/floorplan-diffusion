"""Non-diffusion baseline: the same U-Net, predicting the plan in one pass.

Input: the 7-channel condition (masked one-hot + mask). Output: 6 class
scores ("logits") per pixel. Trained with cross-entropy, mask pixels
weighted more. If the diffusion model can't beat this, diffusion isn't
earning its extra cost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.classes import NUM_CLASSES
from src.models.backbone import build_unet


class UNetInpainter(nn.Module):
    def __init__(self, mcfg):
        super().__init__()
        self.mcfg = mcfg
        self.net = build_unet(mcfg, in_channels=NUM_CLASSES + 1, out_channels=NUM_CLASSES)

    def forward(self, cond):
        # UNet2DModel expects a timestep; this model has none, so always 0.
        t = torch.zeros(len(cond), dtype=torch.long, device=cond.device)
        return self.net(cond, t).sample

    def loss(self, cond, target, mask):
        """Weighted cross-entropy (mask pixels weigh more)."""
        logits = self(cond).float()
        ce = F.cross_entropy(logits, target.argmax(dim=1), reduction="none")  # (B, H, W)
        weight = 1 + (self.mcfg["mask_weight"] - 1) * mask[:, 0]
        return (ce * weight).sum() / weight.sum()

    @torch.no_grad()
    def predict(self, cond, steps=None, generator=None):
        """Label maps (B, H, W). (steps/generator: unused, same API as diffusion.)"""
        return self(cond).argmax(dim=1)
