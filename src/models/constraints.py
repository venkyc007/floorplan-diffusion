"""Structure-aware constraint losses on the predicted clean plan (Day 5).

The diffusion model predicts the clean plan x0 (6 channels in [-1, 1]).
These losses judge its STRUCTURE, not just pixel overlap:

  cldice     wall connectivity: soft clDice between predicted and true
             boundary (wall + door + window). clDice compares SKELETONS
             (1-px centre lines), so a small gap that cuts a wall in two
             costs a lot, while a wall drawn 1 px thicker costs little.
  openings   doors/windows must sit in walls. "Wall support" spreads from
             wall pixels into touching door/window pixels, step by step
             (geodesic dilation), so a door attached to a wall is supported
             along its whole length; a floating door gets no support.
             Score = unsupported opening probability per region pixel.
  closure    rooms must be closed: room probability right next to outdoor
             or background probability is a leak (per region pixel).
  alignment  walls should be horizontal/vertical: from Sobel edge strengths
             gx, gy of the boundary map, |gx*gy| is ~0 for axis-aligned
             edges and large for diagonal ones; normalised by the edge
             energy (gx^2 + gy^2)/2, so the score is 0 (axis-aligned) .. 1 (45 deg).

All losses:
  - use class probabilities p = softmax(x0 / temperature) over the 6 channels,
  - are computed only in the mask region grown by `dilate_px` (default 3),
  - return one value per sample (shape (B,)); the caller weights them by
    noise level: only t_min <= t < t_max (configurable). The run-1 model's
    x0 guess is already structurally perfect below t ~ 600, so the useful
    signal is at high noise (see docs/decisions.md).
"""

import torch
import torch.nn.functional as F

from src.data.classes import BACKGROUND, DOOR, OUTDOOR, ROOM, WALL, WINDOW

EPS = 1e-6
NAMES = ["cldice", "openings", "closure", "alignment"]


# ------------------------------------------------------------ helpers

def probabilities(x0, temperature):
    """Predicted x0 (B, 6, H, W) in [-1, 1] -> class probabilities."""
    return F.softmax(x0 / temperature, dim=1)


def grow(x, px):
    """Grow (dilate) a (B, 1, H, W) map by px pixels in every direction."""
    return F.max_pool2d(x, 2 * px + 1, stride=1, padding=px)


def area_sum(x):
    """Sum over the image -> one value per sample."""
    return x.flatten(1).sum(1)


# ------------------------------------------------- soft skeleton (clDice)

def soft_erode(x):
    """Soft erosion with a 3x3 cross (min over vertical and horizontal pairs)."""
    a = -F.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0))
    b = -F.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1))
    return torch.min(a, b)


def soft_dilate(x):
    return F.max_pool2d(x, 3, stride=1, padding=1)


def soft_skeleton(x, iterations):
    """Differentiable skeleton (Shit et al., clDice, CVPR 2021)."""
    skel = F.relu(x - soft_dilate(soft_erode(x)))
    for _ in range(iterations):
        x = soft_erode(x)
        delta = F.relu(x - soft_dilate(soft_erode(x)))
        skel = skel + F.relu(delta - skel * delta)
    return skel


# ------------------------------------------------------------- the losses

def cldice_loss(p, target, region, iterations):
    """1 - clDice between predicted and true boundary, inside `region`."""
    pred_b = p[:, [WALL, DOOR, WINDOW]].sum(1, keepdim=True)
    true_b = target[:, [WALL, DOOR, WINDOW]].sum(1, keepdim=True)
    skel_p, skel_t = soft_skeleton(pred_b, iterations), soft_skeleton(true_b, iterations)
    t_prec = (area_sum(skel_p * true_b * region) + EPS) / (area_sum(skel_p * region) + EPS)
    t_sens = (area_sum(skel_t * pred_b * region) + EPS) / (area_sum(skel_t * region) + EPS)
    return 1 - 2 * t_prec * t_sens / (t_prec + t_sens)


def openings_loss(p, region, iterations):
    """Unsupported door/window probability per region pixel.

    Normalised by the REGION size, not by the amount of door/window: after
    the softmax every pixel keeps a tiny leftover probability of every
    class, so a region with no real opening would otherwise divide a tiny
    unsupported amount by a tiny total and score ~1 for nothing."""
    openings = p[:, [DOOR, WINDOW]].sum(1, keepdim=True)
    support = soft_dilate(p[:, [WALL]])               # pixels touching a wall (3x3)
    for _ in range(iterations):                       # spread support through openings
        support = torch.max(support, soft_dilate(support * openings))
    support = support.clamp(max=1)
    return area_sum(openings * (1 - support) * region) / (area_sum(region) + EPS)


def closure_loss(p, region):
    """Room probability touching outdoor/background (3x3), per region pixel
    (normalised by the region size, for the same reason as openings_loss)."""
    room = p[:, [ROOM]]
    outside = soft_dilate(p[:, [OUTDOOR]] + p[:, [BACKGROUND]])
    return area_sum(room * outside * region) / (area_sum(region) + EPS)


SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def alignment_loss(p, region):
    """Diagonal edge share of the boundary map: |gx*gy| / ((gx^2 + gy^2)/2)."""
    b = p[:, [WALL, DOOR, WINDOW]].sum(1, keepdim=True)
    kx = SOBEL_X.to(b)[None, None]
    gx = F.conv2d(b, kx, padding=1)
    gy = F.conv2d(b, kx.transpose(2, 3), padding=1)
    diagonal = area_sum((gx * gy).abs() * region)
    energy = area_sum((gx ** 2 + gy ** 2) / 2 * region)
    return diagonal / (energy + EPS)


# -------------------------------------------------------------- combined

def constraint_losses(x0_pred, target, mask, ccfg):
    """All enabled constraint losses, one value per sample each: {name: (B,)}."""
    p = probabilities(x0_pred.float(), ccfg["temperature"])
    region = grow(mask.float(), ccfg["dilate_px"])
    out = {}
    if ccfg["cldice"]["use"]:
        out["cldice"] = cldice_loss(p, target, region, ccfg["cldice"]["iterations"])
    if ccfg["openings"]["use"]:
        out["openings"] = openings_loss(p, region, ccfg["openings"]["iterations"])
    if ccfg["closure"]["use"]:
        out["closure"] = closure_loss(p, region)
    if ccfg["alignment"]["use"]:
        out["alignment"] = alignment_loss(p, region)
    return out


def noise_weight(t, ccfg):
    """1 for noise levels t_min <= t < t_max, else 0."""
    return ((t >= ccfg.get("t_min", 0)) & (t < ccfg["t_max"])).float()
