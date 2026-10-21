"""Conditional diffusion model for inpainting (Palette-style).

Idea: during training we take the clean plan, add a random amount of noise,
and teach the network to recover the clean plan from the noisy one, given
the condition (the corrupted plan + mask). To generate, we start from pure
noise and clean it up in 50 steps (DDIM), each step guided by the condition.

Shapes (B = batch size):
  x0     (B, 6, H, W)  clean one-hot plan scaled to [-1, 1]
  cond   (B, 7, H, W)  model input: masked one-hot (0/1, zero in the mask) + mask
  x_t    (B, 6, H, W)  x0 with noise level t
  net input = concat(x_t, cond) -> 13 channels; net output = predicted x0 (6 ch)

During sampling the known region is kept fixed: after every step, pixels
outside the mask are overwritten with the known plan noised to the same
level as the current sample. At the end we take the argmax per pixel.
"""

import torch
import torch.nn as nn
from diffusers import DDIMScheduler, DDPMScheduler

from src.stage1.data.classes import NUM_CLASSES
from src.stage1.models.backbone import build_unet
from src.stage1.models.constraints import constraint_losses, noise_weight


class DiffusionInpainter(nn.Module):
    def __init__(self, mcfg):
        super().__init__()
        self.mcfg = mcfg
        self.net = build_unet(mcfg, in_channels=2 * NUM_CLASSES + 1, out_channels=NUM_CLASSES)
        schedule = dict(num_train_timesteps=mcfg["num_train_timesteps"],
                        beta_schedule=mcfg["beta_schedule"],
                        prediction_type="sample",   # the network predicts x0
                        clip_sample=True)           # keep predicted x0 in [-1, 1]
        self.train_scheduler = DDPMScheduler(**schedule)
        self.sample_scheduler = DDIMScheduler(**schedule, set_alpha_to_one=True)

    def forward(self, x_t, cond, t):
        return self.net(torch.cat([x_t, cond], dim=1), t).sample

    def loss(self, cond, target, mask):
        """Weighted MSE between predicted and true x0 (mask pixels weigh more),
        plus the structural constraint losses if `constraints` is configured
        and enabled (src/stage1/models/constraints.py). The parts are kept in
        self.last_parts so training can log them separately."""
        x0 = target * 2 - 1
        noise = torch.randn_like(x0)
        t = torch.randint(0, self.mcfg["num_train_timesteps"], (len(x0),), device=x0.device)
        x_t = self.train_scheduler.add_noise(x0, noise, t)
        pred = self(x_t, cond, t)
        weight = 1 + (self.mcfg["mask_weight"] - 1) * mask      # (B, 1, H, W)
        err = ((pred.float() - x0) ** 2) * weight
        total = err.sum() / (weight.sum() * NUM_CLASSES)
        self.last_parts = {"mse": total.detach()}

        ccfg = self.mcfg.get("constraints")
        if ccfg and ccfg.get("enabled"):
            per_t = noise_weight(t, ccfg)                         # 0 for noisy steps (t >= t_max)
            for name, values in constraint_losses(pred, target, mask, ccfg).items():
                term = (values * per_t).mean()
                self.last_parts[name] = term.detach()
                total = total + ccfg[name]["weight"] * term
        return total

    @torch.no_grad()
    def predict(self, cond, steps=None, generator=None, eta=None):
        """Generate label maps (B, H, W) for a batch of conditions (DDIM).

        eta: 0 = deterministic DDIM, 1 = stochastic (fresh noise each step).
        Default from the config (`ddim_eta`, 1.0): the run-1 diagnostic found
        stochastic DDIM 50 best (tolerant mIoU 0.753 vs 0.728 for eta 0) and
        structurally cleanest."""
        eta = self.mcfg.get("ddim_eta", 1.0) if eta is None else eta
        sched = self.sample_scheduler
        sched.set_timesteps(steps or self.mcfg["ddim_steps"], device=cond.device)
        mask = cond[:, -1:]                                     # 1 = unknown
        known = cond[:, :NUM_CLASSES] * 2 - 1                   # valid outside the mask

        def noised_known(t):
            noise = torch.randn(known.shape, generator=generator, device=cond.device)
            return sched.add_noise(known, noise, t.reshape(1).expand(len(known)))

        timesteps = sched.timesteps
        x = torch.randn(known.shape, generator=generator, device=cond.device)
        x = torch.where(mask.bool(), x, noised_known(timesteps[0]))
        for i, t in enumerate(timesteps):
            pred_x0 = self(x, cond, t.expand(len(x)))
            x = sched.step(pred_x0.float(), t, x, eta=eta, generator=generator).prev_sample
            # Known region: noised to the NEXT step's level, or clean at the end.
            outside = noised_known(timesteps[i + 1]) if i + 1 < len(timesteps) else known
            x = torch.where(mask.bool(), x, outside)
        return x.argmax(dim=1)
