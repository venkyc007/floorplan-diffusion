"""Optional clean-up of a predicted plan after sampling (Day 6).

Targets dangling wall stubs: dead-end walls (as defined in
src/stage1/eval/arch_metrics.py: a wall end with no door, window, outdoor or
background nearby) inside the mask. For each one:
  1. EXTEND: if another wall lies within EXTEND_PX px straight ahead of the
     stub's end, fill the gap with wall (the stub was meant to join it);
  2. otherwise REMOVE it if it is shorter than N px (a stray stroke); its
     pixels become the most common non-wall class around them.
Pixels outside the mask are NEVER changed (they are known).

    from src.stage1.postprocess import clean
    fixed = clean(pred, mask, n=6)
"""

import numpy as np
from skimage.morphology import skeletonize

from src.stage1.data.classes import BACKGROUND, DOOR, OUTDOOR, WALL, WINDOW
from src.stage1.eval.arch_metrics import END_RADIUS, skeleton_endpoints, window_of

EXTEND_PX = 4
EIGHT_STEPS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]


def trace_branch(skel, start, limit):
    """Follow the skeleton from an end-point until a junction (or `limit`
    steps). Returns the visited pixels (end-point first)."""
    path, prev, cur = [start], None, start
    for _ in range(limit):
        r, c = cur
        nbrs = [(r + dr, c + dc) for dr, dc in EIGHT_STEPS
                if 0 <= r + dr < skel.shape[0] and 0 <= c + dc < skel.shape[1]
                and skel[r + dr, c + dc] and (r + dr, c + dc) != prev]
        if len(nbrs) != 1:          # junction (several ways on) or end of line
            break
        prev, cur = cur, nbrs[0]
        path.append(cur)
    return path


def is_dead_end(label, r, c):
    return not np.isin(window_of(label, r, c, END_RADIUS), [DOOR, WINDOW, OUTDOOR, BACKGROUND]).any()


def try_extend(out, mask, path):
    """Fill a gap of <= EXTEND_PX px ahead of the stub end if a wall is there."""
    if len(path) < 2:
        return False
    (r, c), (r2, c2) = path[0], path[min(2, len(path) - 1)]
    dr, dc = int(np.sign(r - r2)), int(np.sign(c - c2))   # direction pointing out of the stub
    if dr and dc:                                         # diagonal stub: leave it
        return False
    # The centre line stops ~1 px short of the wall's real end: first step
    # past the stub's own wall pixels, then measure the gap to the next wall.
    k = 1
    while k <= 3 and 0 <= r + k * dr < out.shape[0] and 0 <= c + k * dc < out.shape[1]             and out[r + k * dr, c + k * dc] == WALL:
        k += 1
    gap = []
    for k in range(k, k + EXTEND_PX + 1):
        rr, cc = r + k * dr, c + k * dc
        if not (0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]):
            return False
        if out[rr, cc] == WALL:
            break
        gap.append((rr, cc))
    else:
        return False
    if not gap or len(gap) > EXTEND_PX:
        return False
    # Fill with the stub's thickness: the wall pixels across the direction at the end.
    across = (0, 1) if dr else (1, 0)
    er, ec = gap[0][0] - dr, gap[0][1] - dc            # the stub's last wall pixel
    offsets = [j for j in range(-2, 3) if 0 <= er + j * across[0] < out.shape[0]
               and 0 <= ec + j * across[1] < out.shape[1] and out[er + j * across[0], ec + j * across[1]] == WALL]
    for rr, cc in gap:
        for j in offsets:
            y, x = rr + j * across[0], cc + j * across[1]
            if mask[y, x]:
                out[y, x] = WALL
    return True


def remove_stub(out, mask, path):
    """Turn the stub's wall pixels (within 1 px of its centre line, minus the
    junction end) into the most common non-wall class nearby."""
    band = np.zeros(out.shape, bool)
    for r, c in path[:-1] if len(path) > 1 else path:
        band[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
    band &= (out == WALL) & mask
    for r, c in np.argwhere(band):
        around = window_of(out, r, c, 2).ravel()
        around = around[around != WALL]
        if len(around):
            vals, counts = np.unique(around, return_counts=True)
            out[r, c] = vals[counts.argmax()]


def clean(pred, mask, n):
    """Extend or remove dead-end wall stubs inside the mask (see module doc)."""
    out = pred.copy()
    skel = skeletonize(out == WALL)
    for r, c in skeleton_endpoints(skel):
        if not mask[r, c] or not is_dead_end(out, r, c):
            continue
        path = trace_branch(skel, (r, c), limit=max(n, EXTEND_PX) + 2)
        if try_extend(out, mask, path):
            continue
        if len(path) < n:
            remove_stub(out, mask, path)
    return np.where(mask, out, pred)          # never change known pixels
