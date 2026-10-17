"""16-class FILL models: Palette-style x0-prediction diffusion and a one-pass U-Net.

Same design as Stage 1 (src/models/diffusion.py, unet_baseline.py), generalised to 16
classes, with the Stage 2 constraint losses (src/stage2/constraints) built in:

    L = L_rec + g(t) * ramp(step) * sum_i w_i * L_i          (spec section 2)

  L_rec   diffusion: weighted MSE on x0 (hole pixels x mask_weight); U-Net: weighted CE
  P       softmax(x0_pred / 0.2) (diffusion) or softmax(logits) (U-Net)
  g(t)    diffusion only: 1 for t <= t_full (400), linear to 0 at t_zero (800)
  ramp    0 -> 1 linearly over ramp_steps (1,000) to protect the warm-started weights
  w_i     per-constraint weights from the config (calibrated by gradient size, spec section 3)

Shapes: cond (B, 17, H, W) = one-hot with the hole zeroed (16) + mask (1);
target (B, 16, H, W) one-hot; mask (B, 1, H, W), 1 = hole.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler

from src.models.backbone import build_unet
from src.stage2.constraints.constraints import Ctx, constraint_loss, load_constants
from src.stage2.data.classes16 import C16

N = C16.n
CONSTANTS = load_constants()     # read once (configs/stage2/constraint_constants.yaml)


def gate(t, ccfg):
    """g(t): 1 for t <= t_full, linear to 0 at t_zero (per sample)."""
    lo, hi = ccfg["gate"]["t_full"], ccfg["gate"]["t_zero"]
    return ((hi - t.float()) / (hi - lo)).clamp(0, 1)


def ramp(step, ccfg):
    return min(1.0, step / max(1, ccfg.get("ramp_steps", 1))) if step is not None else 1.0


def constraint_terms(P, target, mask, ccfg, per_sample_weight=None):
    """{name: scalar loss} for every active constraint (fp32). per_sample_weight: (B,) e.g. g(t)."""
    ctx = Ctx(hole=mask.float(), Y=target.float(), known=target.float() * (1 - mask.float()),
              const=CONSTANTS, K=ccfg.get("K", 96))
    out = {}
    for name in ccfg["active"]:
        mode = ccfg.get("modes", {}).get(name)
        l = constraint_loss(name, P, ctx, mode)                 # (B,)
        if per_sample_weight is not None:
            l = l * per_sample_weight
        out[name] = l.mean()
    return out


class Diffusion16(nn.Module):
    def __init__(self, mcfg):
        super().__init__()
        self.mcfg = mcfg
        self.net = build_unet(mcfg, in_channels=2 * N + 1, out_channels=N)
        schedule = dict(num_train_timesteps=mcfg["num_train_timesteps"], beta_schedule=mcfg["beta_schedule"],
                        prediction_type="sample", clip_sample=True)
        self.train_scheduler = DDPMScheduler(**schedule)
        self.sample_scheduler = DDIMScheduler(**schedule, set_alpha_to_one=True)

    def forward(self, x_t, cond, t):
        return self.net(torch.cat([x_t, cond], dim=1), t).sample

    def loss(self, cond, target, mask, step=None, constraints=True, t=None):
        x0 = target * 2 - 1
        noise = torch.randn_like(x0)
        if t is None:
            t = torch.randint(0, self.mcfg["num_train_timesteps"], (len(x0),), device=x0.device)
        x_t = self.train_scheduler.add_noise(x0, noise, t)
        pred = self(x_t, cond, t)
        weight = 1 + (self.mcfg["mask_weight"] - 1) * mask
        total = (((pred.float() - x0) ** 2) * weight).sum() / (weight.sum() * N)
        self.last_parts = {"rec": total.detach()}
        self.last_pred = pred
        ccfg = self.mcfg.get("constraints")
        if constraints and ccfg and ccfg.get("enabled"):
            P = F.softmax(pred.float() / ccfg["temperature"], dim=1)
            r = ramp(step, ccfg)
            for name, term in constraint_terms(P, target, mask, ccfg, gate(t, ccfg)).items():
                self.last_parts[name] = term.detach()
                total = total + r * ccfg["weights"].get(name, 0.0) * term
        return total

    @torch.no_grad()
    def predict(self, cond, steps=None, generator=None, eta=None):
        """Stochastic DDIM with the known-region reset (as in Stage 1) -> label maps (B, H, W)."""
        eta = self.mcfg.get("ddim_eta", 1.0) if eta is None else eta
        sched = self.sample_scheduler
        sched.set_timesteps(steps or self.mcfg["ddim_steps"], device=cond.device)
        mask = cond[:, -1:]
        known = cond[:, :N] * 2 - 1

        def noised_known(t):
            noise = torch.randn(known.shape, generator=generator, device=cond.device)
            return sched.add_noise(known, noise, t.reshape(1).expand(len(known)))

        ts = sched.timesteps
        x = torch.where(mask.bool(), torch.randn(known.shape, generator=generator, device=cond.device),
                        noised_known(ts[0]))
        for i, t in enumerate(ts):
            pred_x0 = self(x, cond, t.expand(len(x)))
            x = sched.step(pred_x0.float(), t, x, eta=eta, generator=generator).prev_sample
            x = torch.where(mask.bool(), x, noised_known(ts[i + 1]) if i + 1 < len(ts) else known)
        return x.argmax(dim=1)


class UNet16(nn.Module):
    def __init__(self, mcfg):
        super().__init__()
        self.mcfg = mcfg
        self.net = build_unet(mcfg, in_channels=N + 1, out_channels=N)

    def forward(self, cond):
        t = torch.zeros(len(cond), dtype=torch.long, device=cond.device)
        return self.net(cond, t).sample

    def loss(self, cond, target, mask, step=None, constraints=True, t=None):
        logits = self(cond).float()
        ce = F.cross_entropy(logits, target.argmax(dim=1), reduction="none")
        weight = 1 + (self.mcfg["mask_weight"] - 1) * mask[:, 0]
        total = (ce * weight).sum() / weight.sum()
        self.last_parts = {"rec": total.detach()}
        self.last_pred = logits
        ccfg = self.mcfg.get("constraints")
        if constraints and ccfg and ccfg.get("enabled"):
            P = F.softmax(logits, dim=1)
            r = ramp(step, ccfg)
            for name, term in constraint_terms(P, target, mask, ccfg).items():
                self.last_parts[name] = term.detach()
                total = total + r * ccfg["weights"].get(name, 0.0) * term
        return total

    @torch.no_grad()
    def predict(self, cond, steps=None, generator=None):
        return self(cond).argmax(dim=1)


def build16(mcfg):
    return {"diffusion": Diffusion16, "unet": UNet16}[mcfg["kind"]](mcfg)
