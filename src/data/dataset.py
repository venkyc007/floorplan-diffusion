"""PyTorch Dataset for training: a NEW random corruption every time.

Each time the training loop asks for sample i, we:
  1. take clean plan i,
  2. rotate/flip it randomly (one of 8 variants) - free extra variety,
  3. remove something random from it (configs/corruption.yaml -> train):
       80%: one part, type drawn with type_probs (door/window/wall/room/region)
       20%: 2-3 parts of randomly drawn types at once (union of their masks)
  4. return (input, target, mask) as float tensors:
       input  (7, 128, 128)  6 one-hot channels, all zero inside the mask, + mask channel
       target (6, 128, 128)  the clean one-hot plan
       mask   (1, 128, 128)  1 = region the model must fill in

Because the corruption is drawn fresh each time, the model never sees the
exact same training example twice.

Usage:
    ds = CorruptionDataset("data/processed/train.npz", load_config("corruption"))
    x, y, m = ds[0]
"""

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.classes import NUM_CLASSES
from src.data.corruptions import TYPES, corrupt, corrupt_multi, to_model_input
from src.data.dedup import dihedral_variants


def load_labels(npz_path, chunk=1000):
    """Load a split's one-hot masks and turn them into label maps (N, H, W) uint8.

    Done in chunks: label maps need 6x less memory than one-hot arrays.
    """
    onehot = np.load(npz_path)["masks"]
    labels = np.empty((len(onehot),) + onehot.shape[2:], np.uint8)
    for s in range(0, len(onehot), chunk):
        labels[s:s + chunk] = onehot[s:s + chunk].argmax(axis=1)
    return labels


class CorruptionDataset(Dataset):
    def __init__(self, npz_path, cfg, augment=True):
        self.labels = load_labels(npz_path)
        self.cfg = cfg
        self.augment = augment
        train = cfg["train"]
        self.types = list(train["type_probs"])
        probs = np.array([train["type_probs"][t] for t in self.types], float)
        self.probs = probs / probs.sum()
        # How often a requested type had nothing to remove (another type
        # was used instead). Counted per worker process.
        self.skips = {t: 0 for t in TYPES}
        self._rng, self._pid = None, None

    def __len__(self):
        return len(self.labels)

    def rng(self):
        """A random generator per worker process, seeded from PyTorch's seed,
        so torch.manual_seed() makes runs repeatable and workers differ."""
        if self._pid != os.getpid():
            # One OpenCV thread per process: with several DataLoader workers,
            # OpenCV's own threads would otherwise fight over the CPU cores.
            cv2.setNumThreads(1)
            self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
            self._pid = os.getpid()
        return self._rng

    def draw_types(self, n):
        return list(self.rng().choice(self.types, size=n, p=self.probs))

    def corrupt_once(self, label):
        """One random corruption; falls back to other types if needed."""
        rng, train = self.rng(), self.cfg["train"]
        if rng.random() < train["multi_prob"]:
            n = rng.integers(train["multi_count"][0], train["multi_count"][1] + 1)
            result = corrupt_multi(label, self.draw_types(n), rng, self.cfg)
            if result is not None:
                return result
        first = self.draw_types(1)[0]
        for ctype in [first] + [t for t in rng.permutation(self.types) if t != first]:
            result = corrupt(label, ctype, rng, self.cfg)
            if result is not None:
                return result
            self.skips[ctype] += 1
        raise RuntimeError("plan has nothing that can be removed")

    def __getitem__(self, i):
        label = self.labels[i]
        if self.augment:
            label = np.ascontiguousarray(dihedral_variants(label)[self.rng().integers(8)])
        mask, corrupted, _ = self.corrupt_once(label)
        x = to_model_input(corrupted, mask)
        y = (np.arange(NUM_CLASSES)[:, None, None] == label[None]).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask[None].astype(np.float32))
