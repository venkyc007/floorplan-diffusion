"""Wall-continuation candidates ("rays"): where would the walls go if the ones we can see simply continued?

    free, junction = continuation_rays(label)        # two bool maps, same shape as the 16-class map

A straight wall that stops in the open is the strongest hint that a wall is missing: when a room's walls are left
out of a drawing, the walls of the neighbouring rooms still run up to the missing room's corners and end there.
For every end of a straight wall run (horizontal or vertical, at least MIN_RUN px long) a ray is cast onward
through free space (rooms, fixtures) until it meets another wall:
  free      the run ends in the open (a stub: no crossing wall at its end)          - strong candidate
  junction  the run ends at a crossing wall (T or L junction) and the ray continues on the far side - weak candidate
A ray is dropped if it leaves the building (reaches outdoor / background or the map edge), if it is longer than
MAX_LEN px, or if it crosses a door, window or railing (that gap is an opening, not a missing wall).
The rays are INPUT FEATURES for the repair models (they decide which candidates are real) and the exact line a
predicted wall can be snapped to; they are computed from the parser's map only.
"""

import numpy as np

from src.stage2.data.classes16 import C16

WALL = C16.index["wall"]
OPENINGS = [C16.index[c] for c in ("door", "window", "railing")]
OUTSIDE = C16.group("OUT")
MIN_RUN = 7               # px: shorter runs are a wall's thickness, not its direction
CROSS = 7                 # px: a crossing wall at the run's end is at least this long across
MAX_LEN = 160             # px: longest ray (maps are ~150-400 px)
MIN_LEN = 2               # px: shorter gaps are parser speckle


def run_lengths(mask):
    """For every True pixel: the length of the horizontal run of True pixels it belongs to (0 elsewhere)."""
    h, w = mask.shape
    out = np.zeros((h, w), np.int32)
    m = mask.astype(np.int8)
    pad = np.zeros((h, 1), np.int8)
    d = np.diff(np.concatenate([pad, m, pad], 1), axis=1)          # +1 at run starts, -1 after run ends
    for y in range(h):
        starts, ends = np.nonzero(d[y] == 1)[0], np.nonzero(d[y] == -1)[0]
        for s, e in zip(starts, ends):
            out[y, s:e] = e - s
    return out


def _rays_horizontal(label):
    """Rays along rows (to the right and to the left of horizontal run ends) -> (free, junction) bool maps."""
    wall = label == WALL
    hlen, vlen = run_lengths(wall), run_lengths(wall.T).T
    blocked = np.isin(label, OPENINGS)
    outside = np.isin(label, OUTSIDE)
    h, w = label.shape
    free, junction = np.zeros((h, w), bool), np.zeros((h, w), bool)
    ys, xs = np.nonzero(wall & (hlen >= MIN_RUN))
    for y, x in zip(ys, xs):
        for step in (1, -1):
            nx = x + step
            if not (0 <= nx < w) or wall[y, nx]:
                continue                                           # not an end of the run in this direction
            end = nx
            while 0 <= end < w and not wall[y, end] and abs(end - x) <= MAX_LEN:
                end += step
            if not (0 <= end < w) or not wall[y, end]:
                continue                                           # ran off the map or too long: no wall to meet
            lo, hi = (nx, end) if step == 1 else (end + 1, nx + 1)
            if hi - lo < MIN_LEN or blocked[y, lo:hi].any() or outside[y, lo:hi].any():
                continue
            (junction if vlen[y, x] >= CROSS else free)[y, lo:hi] = True
    return free, junction


def continuation_rays(label):
    """(free, junction): bool maps of the ray pixels of both kinds, horizontal and vertical rays together."""
    fh, jh = _rays_horizontal(label)
    fv, jv = _rays_horizontal(np.ascontiguousarray(label.T))
    free, junction = fh | fv.T, jh | jv.T
    return free, junction & ~free
