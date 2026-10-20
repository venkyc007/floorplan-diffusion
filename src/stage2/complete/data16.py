"""Completion-set plans (16-class label maps, walls 3 px) and random training windows.

The data kernel writes completion_<split>.zip with one PNG per plan (uint8 class ids).
Plans are small (median 248 px long side), so they are kept in memory as uint8 arrays.
"""

import zipfile

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.stage2.complete.windows import TYPES, model_inputs, sample_mixed_window, sample_window


def load_plans(zip_path, limit=None, seed=None):
    """[(plan id, label map)] from a completion zip (optionally a random subset of `limit`)."""
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".png"))
        if limit:
            rng = np.random.default_rng(seed)
            names = [names[i] for i in sorted(rng.choice(len(names), min(limit, len(names)), replace=False))]
        return [(n[:-4], cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_UNCHANGED)) for n in names]


def fixed_windows(plans, n, seed=0, win=128, types=TYPES, large_frac=0.0):
    """n reproducible windows (labels, masks, types, plan ids) around random typed deletions
    (large_frac > 0: that share of large holes, windows.sample_mixed_window)."""
    rng = np.random.default_rng(seed)
    labs, masks, kinds, ids = [], [], [], []
    while len(labs) < n:
        pid, lab = plans[rng.integers(len(plans))]
        w = sample_mixed_window(lab, rng, win, large_frac) if large_frac else sample_window(lab, rng, win, types)
        if w is None:
            continue
        labs.append(w[0])
        masks.append(w[1])
        kinds.append(w[2])
        ids.append(pid)
    return np.stack(labs), np.stack(masks), kinds, ids


class WindowSet(Dataset):
    """Infinite-style dataset of fresh random windows (one per request) -> (cond, target, mask)."""

    def __init__(self, plans, win=128, length=10**7, large_frac=0.0):
        self.plans, self.win, self.length, self.large_frac = plans, win, length, large_frac
        self.rng = None

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.rng is None:
            self.rng = np.random.default_rng(torch.initial_seed() % 2**32)
        while True:
            lab = self.plans[self.rng.integers(len(self.plans))][1]
            w = (sample_mixed_window(lab, self.rng, self.win, self.large_frac) if self.large_frac
                 else sample_window(lab, self.rng, self.win))
            if w is not None:
                cond, target, mask = model_inputs(w[0][None], w[1][None])
                return cond[0], target[0], mask[0]
