"""Parser training / validation data from the parser set (greyscale drawing + 16-class labels).

Folder layout (the parser_<split>.zip of the data kernel, extracted):
    <dir>/<plan>_img.png   greyscale drawing, median wall 8 px
    <dir>/<plan>_lbl.png   uint8 class ids (255 = ignore: room types of untyped plans)

Training samples are random 512 x 512 crops at a random drawing scale (0.6-1.6x):
half are centred on a pixel of a rare class (door, window, railing, fixtures) so the
small components are seen often; the rest on a random pixel. Then 90-degree rotation,
flip, and the photometric augmentations (augment.py).
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.stage2.data.classes16 import C16
from src.stage2.data.rasterise import whiten_padding
from src.stage2.parse import augment

BACKGROUND = C16.index["background"]


def list_plans(folder):
    return sorted(p.name[: -len("_img.png")] for p in Path(folder).glob("*_img.png"))


def read_pair(folder, plan):
    """Drawing + labels. Baked-in black padding is turned white here as well (the v3 data were
    built with a padding rule that missed padding blocks joined by a thin line, e.g. plan 11784)."""
    img = cv2.imread(str(Path(folder) / f"{plan}_img.png"), cv2.IMREAD_GRAYSCALE)
    lbl = cv2.imread(str(Path(folder) / f"{plan}_lbl.png"), cv2.IMREAD_UNCHANGED)
    img, _ = whiten_padding(img, lbl)
    return img, lbl


def crop_around(img, lbl, cy, cx, size, pad_img):
    """size x size window centred at (cy, cx); outside the drawing = white paper / background."""
    h, w = img.shape
    y0, x0 = cy - size // 2, cx - size // 2
    out_i = np.full((size, size), pad_img, np.uint8)
    out_l = np.full((size, size), BACKGROUND, np.uint8)
    sy0, sx0, sy1, sx1 = max(0, y0), max(0, x0), min(h, y0 + size), min(w, x0 + size)
    if sy1 > sy0 and sx1 > sx0:
        out_i[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
        out_l[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = lbl[sy0:sy1, sx0:sx1]
    return out_i, out_l


def random_crop(img, lbl, rng, cfg_data, cfg_aug, crop):
    """One augmented (crop x crop) training pair. Crops with less than `min_foreground`
    non-background pixels are rejected and re-drawn (at most `max_tries` draws; the last
    draw is kept if all fail)."""
    for _ in range(cfg_data.get("max_tries", 10)):
        x, y = _one_crop(img, lbl, rng, cfg_data, cfg_aug, crop)
        if (y != BACKGROUND).mean() >= cfg_data.get("min_foreground", 0.0):
            break
    return x, y


def _one_crop(img, lbl, rng, cfg_data, cfg_aug, crop):
    scale = rng.uniform(*cfg_aug["scale"])
    src = max(8, int(round(crop / scale)))          # window in the drawing that becomes `crop` px
    rare = [C16.index[c] for c in cfg_data["rare_classes"]]
    centre = None
    if rng.random() < cfg_data["rare_centre_prob"]:
        present = [c for c in rare if (lbl == c).any()]
        if present:
            ys, xs = np.nonzero(lbl == present[rng.integers(len(present))])
            i = rng.integers(len(ys))
            centre = (int(ys[i]), int(xs[i]))
    if centre is None:
        centre = (int(rng.integers(img.shape[0])), int(rng.integers(img.shape[1])))
    ci, cl = crop_around(img, lbl, *centre, src, cfg_data["pad_image"])
    interp = cv2.INTER_AREA if src > crop else cv2.INTER_LINEAR
    ci = cv2.resize(ci, (crop, crop), interpolation=interp)
    cl = cv2.resize(cl, (crop, crop), interpolation=cv2.INTER_NEAREST)
    ci, cl = augment.rot_flip(ci, cl, rng, cfg_aug["rot90"], cfg_aug["flip"])
    return augment.photometric(ci, rng, cfg_aug), cl


class ParserTrainSet(Dataset):
    """Infinite-style dataset: item i -> a random augmented crop of plan i % n_plans."""

    def __init__(self, folder, cfg, crop=None, length=10**7):
        self.folder, self.cfg = folder, cfg
        self.plans = list_plans(folder)
        self.crop = crop or cfg["data"]["crop"]
        self.length = length
        self.rng = None

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.rng is None:                        # one generator per worker process
            self.rng = np.random.default_rng(torch.initial_seed() % 2**32)
        img, lbl = read_pair(self.folder, self.plans[i % len(self.plans)])
        x, y = random_crop(img, lbl, self.rng, self.cfg["data"], self.cfg["augment"], self.crop)
        return torch.from_numpy(x)[None].float() / 255, torch.from_numpy(y.astype(np.int64))


def pad_to_multiple(img, lbl, m, pad_img, ignore):
    """Pad bottom/right so both sides are multiples of m (padding: paper / ignore)."""
    h, w = img.shape
    H, W = -(-h // m) * m, -(-w // m) * m
    pi = np.full((H, W), pad_img, np.uint8)
    pl = np.full((H, W), ignore, np.uint8)
    pi[:h, :w], pl[:h, :w] = img, lbl
    return pi, pl
