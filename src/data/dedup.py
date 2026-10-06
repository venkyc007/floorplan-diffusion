"""Finding near-duplicate floor plans.

Real-estate listings are often re-posted, so ResPlan contains (almost)
identical plans. ResPlan also contains "augmented" copies that are rotated,
flipped or rescaled versions of other plans. If one copy lands in training
and another in testing, the test score is unfairly high ("leakage").

How we compare two plans:
  1. Take each plan's STRUCTURE: the pixels that are wall, door or window
     (together they trace every wall line of the plan).
  2. Shrink it to a small grid (e.g. 32x32) by averaging. Small shifts of a
     pixel or two then barely matter.
  3. Compare grids with "cosine similarity": 1.0 = identical, 0 = no overlap.
  4. Try all 8 rotations/flips of one plan and keep the best score, so a
     rotated copy still counts as a duplicate. (Rescaled copies are already
     identical, because every plan is scaled to fill the image.)
"""

import numpy as np
import torch

from src.data.classes import DOOR, WALL, WINDOW


def structure_descriptor(label, grid):
    """Label map (H, W) -> small structure grid (grid, grid) of values 0..1."""
    structure = np.isin(label, [WALL, DOOR, WINDOW]).astype(np.float32)
    k = label.shape[0] // grid
    return structure.reshape(grid, k, grid, k).mean(axis=(1, 3))


def dihedral_variants(img):
    """All 8 rotations (0/90/180/270 degrees) and mirror images of a square grid."""
    out = []
    for flipped in (img, img[:, ::-1]):
        for k in range(4):
            out.append(np.rot90(flipped, k))
    return out


def _normalize(x):
    """Scale each row to length 1, so a dot product = cosine similarity."""
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)


def find_similar_pairs(labels, grid, threshold, chunk=256):
    """Return every pair (i, j, similarity) with i < j and similarity >= threshold.

    Compares every plan with every other plan. To fit in memory this is done
    in chunks of `chunk` plans at a time.
    """
    n = len(labels)
    descs = [structure_descriptor(l, grid) for l in labels]
    # base: one row per plan, as stored.  (n, grid*grid)
    base = torch.from_numpy(_normalize(np.stack([d.ravel() for d in descs])))
    # variants: 8 rows per plan, its rotations/flips.  (n, 8, grid*grid)
    variants = torch.from_numpy(_normalize(np.stack(
        [np.stack([v.ravel() for v in dihedral_variants(d)]) for d in descs]
    ).reshape(n * 8, -1)).reshape(n, 8, -1))

    pairs = []
    for start in range(0, n, chunk):
        q = variants[start:start + chunk]                      # (c, 8, D)
        sims = torch.einsum("cvd,nd->cvn", q, base).amax(dim=1)  # best of 8 -> (c, n)
        rows, cols = torch.nonzero(sims >= threshold, as_tuple=True)
        for r, j in zip(rows.tolist(), cols.tolist()):
            i = start + r
            if i < j:  # each pair once, and skip a plan matched with itself
                pairs.append((i, j, float(sims[r, j])))
    return pairs


class UnionFind:
    """Groups items that are linked, directly or through a chain of links.

    If A~B and B~C, then A, B and C all end up in the same group.
    """

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]  # shortcut for speed
            i = self.parent[i]
        return i

    def union(self, i, j):
        self.parent[self.find(i)] = self.find(j)

    def groups(self):
        """Return a list where entry i is the group number of item i."""
        return [self.find(i) for i in range(len(self.parent))]


def group_pairs(n, pairs, threshold):
    """Group plans linked by pairs with similarity >= threshold.

    Returns a list: entry i = group number of plan i.
    """
    uf = UnionFind(n)
    for i, j, s in pairs:
        if s >= threshold:
            uf.union(i, j)
    return uf.groups()
