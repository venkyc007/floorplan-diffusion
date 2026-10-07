"""Finding and removing near-duplicate floor plans.

Real-estate listings are often re-posted, so ResPlan contains (almost)
identical plans. If one copy lands in training and another in testing, the
test score is unfairly high ("leakage").

Comparing every plan with every other plan at full resolution would take far
too long (17,000 x 17,000 pairs), so we work in two stages:

  Stage 1 - find CANDIDATES (fast, rough):
    Shrink each plan's STRUCTURE (wall + door + window pixels) to a small
    grid (e.g. 32x32) and compare grids with "cosine similarity"
    (1.0 = identical). All 8 rotations/flips are tried, so rotated copies
    are found too. Pairs above a low threshold become candidates.

  Stage 2 - CONFIRM candidates (slow, exact):
    Rotate/flip one plan to match the other, then compute the mean
    per-class IoU over the full 128x128 6-channel images.
    IoU ("intersection over union") of one class = pixels where BOTH plans
    have that class / pixels where EITHER plan has it. 1.0 = identical.
    We use a TOLERANT IoU: a pixel counts as matching if the other plan has
    the same class within `tolerance` pixels. Walls and doors are only ~2 px
    thick, so with plain IoU a copy traced 1 px differently loses about half
    its wall/door IoU (near-identical re-posts scored only ~0.8).
"""

import cv2
import numpy as np
import torch

from src.data.classes import BACKGROUND, DOOR, NUM_CLASSES, WALL, WINDOW

# Classes compared in Stage 2. Background is left out: it is mostly empty
# border, so it would make every pair look similar.
IOU_CLASSES = [c for c in range(NUM_CLASSES) if c != BACKGROUND]


# ---------------------------------------------------------------- Stage 1

def structure_descriptor(label, grid):
    """Label map (H, W) -> small structure grid (grid, grid) of values 0..1."""
    structure = np.isin(label, [WALL, DOOR, WINDOW]).astype(np.float32)
    k = label.shape[0] // grid
    return structure.reshape(grid, k, grid, k).mean(axis=(1, 3))


def dihedral_variants(img):
    """All 8 rotations (0/90/180/270 degrees) and mirror images of a square image.

    Variant number v = 4 * flipped + number_of_90_degree_turns.
    """
    out = []
    for flipped in (img, img[:, ::-1]):
        for k in range(4):
            out.append(np.rot90(flipped, k))
    return out


def _normalize(x):
    """Scale each row to length 1, so a dot product = cosine similarity."""
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)


def find_candidates(labels, grid, threshold, chunk=256):
    """Stage 1. Return pairs (i, j, cosine, v) with i < j and cosine >= threshold.

    v = which rotation/flip of plan i matches plan j best.
    Compares every plan with every other plan, `chunk` plans at a time.
    """
    n = len(labels)
    descs = [structure_descriptor(l, grid) for l in labels]
    base = torch.from_numpy(_normalize(np.stack([d.ravel() for d in descs])))
    variants = torch.from_numpy(_normalize(np.stack(
        [np.stack([v.ravel() for v in dihedral_variants(d)]) for d in descs]
    ).reshape(n * 8, -1)).reshape(n, 8, -1))

    pairs = []
    for start in range(0, n, chunk):
        q = variants[start:start + chunk]                       # (c, 8, D)
        sims, best_v = torch.einsum("cvd,nd->cvn", q, base).max(dim=1)  # (c, n)
        rows, cols = torch.nonzero(sims >= threshold, as_tuple=True)
        for r, j in zip(rows.tolist(), cols.tolist()):
            i = start + r
            if i < j:  # each pair once, and skip a plan matched with itself
                pairs.append((i, j, float(sims[r, j]), int(best_v[r, j])))
    return pairs


# ---------------------------------------------------------------- Stage 2

def tolerant_iou(a, b, tolerance=1):
    """Mean tolerant IoU over IOU_CLASSES of two (already aligned) label maps.

    For each class: a pixel of plan a "matches" if plan b has the same class
    within `tolerance` pixels (and the other way round). Then
        IoU = matches / (pixels in a + pixels in b - matches)
    which equals ordinary IoU when tolerance=0.
    Classes absent from BOTH plans are skipped (e.g. no outdoor area at all),
    otherwise they would count as 0/0.
    """
    kernel = np.ones((2 * tolerance + 1,) * 2, np.uint8)
    ious = []
    for c in IOU_CLASSES:
        in_a = (a == c).astype(np.uint8)
        in_b = (b == c).astype(np.uint8)
        total = int(in_a.sum() + in_b.sum())
        if total == 0:
            continue
        # Average of "a near b" and "b near a", so the score is symmetric.
        matches = (np.count_nonzero(in_a & cv2.dilate(in_b, kernel)) +
                   np.count_nonzero(in_b & cv2.dilate(in_a, kernel))) / 2
        ious.append(matches / (total - matches))
    return float(np.mean(ious)) if ious else 0.0


def confirm_candidates(labels, candidates, tolerance=1):
    """Stage 2. Rotate/flip plan i as found in Stage 1, then add the
    full-resolution tolerant IoU to every candidate pair.

    Returns pairs (i, j, cosine, iou).
    """
    out = []
    for k, (i, j, cos, v) in enumerate(candidates):
        aligned = dihedral_variants(labels[i])[v]
        out.append((i, j, cos, tolerant_iou(aligned, labels[j], tolerance)))
        if (k + 1) % 5000 == 0:
            print(f"  checked {k + 1}/{len(candidates)} candidate pairs")
    return out


# ---------------------------------------------------------------- Removal

def remove_duplicates(n, pairs, threshold, priority):
    """Decide which plans to remove. No "chain" removals.

    Plans are visited in `priority` order (best plans first). A plan is
    REMOVED only if it is directly similar (iou >= threshold) to a plan that
    was already KEPT; otherwise it is kept. So every removed plan has a real
    near-copy in the dataset.

    Returns a list: entry i = index of the kept plan that i duplicates
    (or -1 if i is kept), and the iou of that match.
    """
    neighbours = [[] for _ in range(n)]
    for i, j, _, iou in pairs:
        if iou >= threshold:
            neighbours[i].append((j, iou))
            neighbours[j].append((i, iou))

    kept = np.zeros(n, dtype=bool)
    duplicate_of = np.full(n, -1)
    match_iou = np.full(n, np.nan)
    for i in priority:
        matches = [(iou, j) for j, iou in neighbours[i] if kept[j]]
        if matches:
            match_iou[i], duplicate_of[i] = max(matches)  # closest kept plan
        else:
            kept[i] = True
    return duplicate_of, match_iou


# ---------------------------------------------------------------- Grouping

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


def group_pairs(n, links):
    """Group plans linked by the (i, j) pairs in `links`.

    Returns a list: entry i = group number of plan i.
    """
    uf = UnionFind(n)
    for i, j in links:
        uf.union(i, j)
    return uf.groups()
