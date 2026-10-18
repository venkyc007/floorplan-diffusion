"""16-class FILL sampling: hybrid sketch -> DDIM, constraint guidance in x0-space, best-of-N verifier
(spec section 5).

    labels, info = fill(diffusion, cond, sketch=unet_labels, t0=600, n=8, eta_guidance=0.3)

hybrid_sample  The U-Net sketch (argmax labels) is noised to t0 and finished with stochastic DDIM
               (steps <= t0 of a 50-step schedule) with the known-region reset (as in Stage 1).
               Without a sketch: full DDIM from noise.
guidance       At every DDIM step with t <= t_max (600): x0 <- x0 - eta * grad_x0 sum_i w_i L_i(softmax(x0 / 0.2)),
               ABSOLUTE mode, inside M = hole grown 3 px; then the usual DDIM update with the corrected x0.
               No backprop through the network (only through the constraint functions).
verifier       N samples per hole; keep the one with the fewest thresholded violations (connected blobs of
               V > 0.5 inside M for every constraint usable on a window, + R1 for rooms inside the window
               + R3); ties -> highest pixel agreement with the U-Net sketch inside the hole.
C17 is training-only (needs the ground truth) and R4 needs the whole plan, so neither is used here.
"""

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.stage2.constraints import ops
from src.stage2.constraints.constraints import CONST, Ctx, constraint_loss, one_hot, violation_map
from src.stage2.constraints.rules import components, r3_adjacency, room_instances
from src.stage2.data.classes16 import C16
from src.utils import PROJECT_ROOT

N = C16.n
GUIDE = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C18"]


def guidance_weights(path=PROJECT_ROOT / "configs" / "stage2" / "constraint_weights_calibrated.yaml"):
    """Calibrated weights of every constraint usable at inference (incl. the 4 left out of training)."""
    with open(path, encoding="utf-8") as f:
        w = yaml.safe_load(f)["weights"]
    return {k: v for k, v in w.items() if k in GUIDE}


def guide_x0(x0, mask, known, weights, eta, temperature=0.2, K=96):
    """One guidance step on the predicted x0 (B, 16, H, W) in [-1, 1]."""
    with torch.enable_grad():
        x = x0.detach().float().requires_grad_()
        P = F.softmax(x / temperature, dim=1)
        ctx = Ctx(hole=mask, known=known, const=CONST, K=K)
        loss = sum(w * constraint_loss(n, P, ctx, "absolute").sum() for n, w in weights.items())
        g, = torch.autograd.grad(loss, x)
    return (x0 - eta * g * mask).clamp(-1, 1)          # only the hole changes


@torch.no_grad()
def hybrid_sample(model, cond, sketch=None, t0=600, steps=50, eta=1.0, eta_guidance=0.0, t_guide=600,
                  weights=None, generator=None):
    """-> label maps (B, H, W). cond (B, 17, H, W); sketch (B, H, W) labels or None."""
    sched = model.sample_scheduler
    sched.set_timesteps(steps, device=cond.device)
    mask, known01 = cond[:, -1:], cond[:, :N]
    known = known01 * 2 - 1
    weights = weights if weights is not None else guidance_weights()

    def noised(x0, t):
        noise = torch.randn(x0.shape, generator=generator, device=cond.device)
        return sched.add_noise(x0, noise, t.reshape(1).expand(len(x0)))

    ts = sched.timesteps
    if sketch is not None:
        ts = ts[ts <= t0]
        start = one_hot(sketch.to(cond.device)) * 2 - 1
        x = noised(torch.where(mask.bool(), start, known), ts[0])
    else:
        x = torch.where(mask.bool(), torch.randn(known.shape, generator=generator, device=cond.device), noised(known, ts[0]))
    for i, t in enumerate(ts):
        x0 = model(x, cond, t.expand(len(x))).float()
        if eta_guidance > 0 and int(t) <= t_guide:
            x0 = guide_x0(x0, mask, known01, weights, eta_guidance)
        x = sched.step(x0, t, x, eta=eta, generator=generator).prev_sample
        x = torch.where(mask.bool(), x, noised(known, ts[i + 1]) if i + 1 < len(ts) else known)
    return x.argmax(dim=1)


@torch.no_grad()
def violation_count(labels, mask, known01, const=CONST):
    """Thresholded violations per sample: connected blobs of V > 0.5 inside M, summed over the
    window-usable constraints, + R1 (rooms not cut by the window border) + R3. -> (B,) ints."""
    P = one_hot(labels)
    M = ops.dil(mask, 7) > 0
    ctx = Ctx(hole=mask, known=known01, const=const, K=96)
    counts = np.zeros(len(labels), int)
    for name in GUIDE:
        v = (violation_map(name, P, ctx) > 0.5) & M
        for b in range(len(labels)):
            counts[b] += len(components(v[b, 0].cpu().numpy()))
    lab = labels.cpu().numpy()
    m = M[:, 0].cpu().numpy()
    ranges = const.get("room_area", {})
    for b in range(len(lab)):
        for c, inst in room_instances(lab[b]):
            if inst[0].any() or inst[-1].any() or inst[:, 0].any() or inst[:, -1].any() or not (inst & m[b]).any():
                continue
            r = ranges.get(C16.names[c])
            counts[b] += bool(r and not (r[0] <= inst.sum() <= r[1]))
        counts[b] += len(components(r3_adjacency(lab[b], const)[1] & m[b]))
    return counts


@torch.no_grad()
def best_of_n(samples, mask, known01, sketch=None):
    """samples (N, H, W) for ONE hole -> (index of the chosen sample, violation counts, agreements)."""
    n = len(samples)
    counts = violation_count(samples, mask.expand(n, -1, -1, -1), known01.expand(n, -1, -1, -1))
    agree = np.zeros(n)
    if sketch is not None:
        h = mask[0, 0].bool()
        agree = np.array([float((s[h] == sketch[h]).float().mean()) for s in samples])
    order = sorted(range(n), key=lambda i: (counts[i], -agree[i]))
    return order[0], counts, agree


@torch.no_grad()
def fill(model, cond, sketch=None, n=8, t0=600, eta_guidance=0.0, steps=50, generator=None, weights=None):
    """Hybrid sampling + guidance + best-of-N for a batch of holes -> (labels (B, H, W), info list)."""
    out, info = [], []
    for b in range(len(cond)):
        c = cond[b:b + 1].expand(n, -1, -1, -1)
        sk = sketch[b:b + 1].expand(n, -1, -1) if sketch is not None else None
        samples = hybrid_sample(model, c, sk, t0=t0, steps=steps, eta_guidance=eta_guidance,
                                weights=weights, generator=generator)
        i, counts, agree = best_of_n(samples, cond[b:b + 1, -1:], cond[b:b + 1, :N],
                                     sketch[b] if sketch is not None else None)
        out.append(samples[i])
        info.append({"chosen": i, "violations": counts.tolist(), "agreement": agree.round(3).tolist()})
    return torch.stack(out), info
