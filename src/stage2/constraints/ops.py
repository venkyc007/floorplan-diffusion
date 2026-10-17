"""Differentiable morphology operators for the Stage 2 constraints (spec section 0).

All inputs are float maps of shape (B, C, H, W) with values in [0, 1]
(class probabilities or one-hot). Everything is built on max_pool2d, so it is
differentiable and runs on the GPU.

Borders: dilation pads with -inf (nothing outside the image can "grow in"),
and erosion is -dilation(-X), so for erosion the outside counts as +inf
(a room cut by the window border is NOT eroded from the border). This avoids
false violations at the edge of 128-px training windows.

Kernel sizes may be even; the extra pixel goes to the bottom/right.
"""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

NEG = float("-inf")


# ------------------------------------------------------------------ basics

def shift(x, dy, dx):
    """Translate by (dy, dx) pixels with zero padding: out[y, x] = x[y - dy, x - dx]."""
    h, w = x.shape[-2:]
    out = torch.zeros_like(x)
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    if (ys.stop - ys.start) > 0 and (xs.stop - xs.start) > 0:
        out[..., yd, xd] = x[..., ys, xs]
    return out


def _pool(x, kh, kw, mirror=False):
    """Max over a kh x kw window around each pixel (-inf outside the image).

    For even sizes the window covers one more pixel below/right of the centre;
    `mirror` flips that (one more above/left). An opening must dilate with the
    mirrored window of its erosion, or even-sized openings shift by one pixel.
    """
    kh, kw = int(kh), int(kw)
    if kh <= 1 and kw <= 1:
        return x
    if kh > 1 and kw > 1:          # a square max is separable: rows, then columns (same result, much faster)
        return _pool(_pool(x, 1, kw, mirror), kh, 1, mirror)
    t, l = (kh - 1) // 2, (kw - 1) // 2
    if mirror:
        t, l = kh - 1 - t, kw - 1 - l
    x = F.pad(x, (l, kw - 1 - l, t, kh - 1 - t), value=NEG)
    return F.max_pool2d(x, (kh, kw), stride=1)


def dil(x, k):
    """Soft dilation with a k x k square."""
    return _pool(x, k, k)


def ero(x, k):
    """Soft erosion with a k x k square."""
    return -_pool(-x, k, k)


def opn(x, k):
    """Soft opening: removes everything thinner than k (in both directions)."""
    return _pool(ero(x, k), k, k, mirror=True)


def dil_h(x, n):
    return _pool(x, 1, n)


def dil_v(x, n):
    return _pool(x, n, 1)


def ero_h(x, n):
    return -_pool(-x, 1, n)


def ero_v(x, n):
    return -_pool(-x, n, 1)


def opn_h(x, n):
    """Keeps only horizontal runs of length >= n."""
    return _pool(ero_h(x, n), 1, n, mirror=True)


def opn_v(x, n):
    return _pool(ero_v(x, n), n, 1, mirror=True)


# ------------------------------------------------------------------ directional

def _side_max(x, n, before, dim):
    """Max of x over the n pixels strictly before (left/up) or after (right/down) each pixel."""
    n = int(n)
    if dim == -1:
        pad = (n, 0, 0, 0) if before else (0, n, 0, 0)
        p = F.max_pool2d(F.pad(x, pad, value=0.0), (1, n), stride=1)
        return p[..., :, :-1] if before else p[..., :, 1:]
    pad = (0, 0, n, 0) if before else (0, 0, 0, n)
    p = F.max_pool2d(F.pad(x, pad, value=0.0), (n, 1), stride=1)
    return p[..., :-1, :] if before else p[..., 1:, :]


def flank_h(x, n):
    """'Is there x on BOTH sides horizontally within n px': min(max left, max right)."""
    return torch.minimum(_side_max(x, n, True, -1), _side_max(x, n, False, -1))


def flank_v(x, n):
    return torch.minimum(_side_max(x, n, True, -2), _side_max(x, n, False, -2))


def sum_h(x):
    return shift(x, 0, 1) + shift(x, 0, -1)


def sum_v(x):
    return shift(x, 1, 0) + shift(x, -1, 0)


# ------------------------------------------------------------------ flood fill

def _rec_steps(r, passable, steps):
    for _ in range(steps):
        r = torch.minimum(passable, dil(r, 3))
    return r


def rec(seed, passable, k=96, chunk=16):
    """Soft flood fill (geodesic reconstruction) of `seed` through `passable`, k steps of 3x3.

    Every `chunk` steps are wrapped in torch.utils.checkpoint when gradients are
    needed, so memory does not grow with k.
    """
    r = seed * passable
    done = 0
    while done < k:
        n = min(chunk, k - done)
        if torch.is_grad_enabled() and (r.requires_grad or passable.requires_grad):
            r = checkpoint(_rec_steps, r, passable, n, use_reentrant=False)
        else:
            r = _rec_steps(r, passable, n)
        done += n
    return r


# ------------------------------------------------------------------ edges, smooth max, skeleton

_SOBEL = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 4


def sobel_x(x):
    w = _SOBEL.to(x).view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), w, groups=x.shape[1])


def sobel_y(x):
    w = _SOBEL.t().contiguous().to(x).view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), w, groups=x.shape[1])


def smax(x, region, tau=0.1):
    """Smooth max of x over a boolean/0-1 region (per sample): tau * log(mean(exp(x / tau))).

    Lies between the mean and the max (equal to the max when x is constant). 0 if the region is empty.
    """
    region = region.bool()
    z = (x / tau).masked_fill(~region, NEG).flatten(1)
    n = region.flatten(1).sum(1).clamp(min=1)
    out = tau * (torch.logsumexp(z, dim=1) - torch.log(n.to(x.dtype)))
    return torch.where(region.flatten(1).any(1), out, torch.zeros_like(out))


def soft_erode_cross(x):
    a = -F.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0))
    b = -F.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(a, b)


def soft_skeleton(x, iterations=4):
    """Differentiable skeleton (Shit et al., clDice, CVPR 2021), as in Stage 1."""
    def open3(z):
        return F.max_pool2d(soft_erode_cross(z), 3, stride=1, padding=1)
    skel = F.relu(x - open3(x))
    for _ in range(iterations):
        x = soft_erode_cross(x)
        delta = F.relu(x - open3(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def masked_mean(v, region, eps=1e-6):
    """Per-sample mean of v (B,1,H,W) over region (B,1,H,W) -> (B,)."""
    return (v * region).flatten(1).sum(1) / (region.flatten(1).sum(1) + eps)
