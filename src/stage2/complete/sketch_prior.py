"""Type-guided fill (user, 2026-10-19): paint the detected missing component into the U-Net sketch before the
hybrid re-noises it to t0 and runs DDIM, so the diffusion refines a door / window / fixture / wall that is
already there instead of reproducing the plain context.

    sketch = apply_prior(sketch, hole, typ, rules)     # (H, W) labels of one 128 x 128 window, hole bool

Sizes: median component sizes of the tuning plans (configs/stage2/sketch_prior.yaml, completion scale).
  door     on the hole's wall line (the sketch's walls inside the hole, closed along their direction):
           a gap in that line (R5 / door-ink holes: the gap itself) becomes door; otherwise a door of median
           length across the wall's thickness. With a flag point (user, 2026-10-20: the box centre of an R5 /
           ink flag, the detector's heat peak) the wall line is the one nearest that point and the door is
           centred on it; without one (C8-only flags: a whole room) it is centred on the hole's longest wall run.
  window   the same on an EXTERIOR wall (wall pixels next to outdoor / background); median window length.
  fixture  a median-size box of that fixture against the wall nearest to the hole centre, inside the room.
  wall     the wall line closed across its gap (straight segment between the two wall ends); a measured gap
           mask (pipeline gap rule, src/stage2/detect/flag_gap.py) is painted as wall directly.
Anything else (room, railing, manual) is left as the U-Net drew it.
"""

import cv2
import numpy as np
import yaml

from src.stage2.data.classes16 import C16
from src.utils import PROJECT_ROOT

I = C16.index
SIZES = yaml.safe_load((PROJECT_ROOT / "configs" / "stage2" / "sketch_prior.yaml").read_text(encoding="utf-8"))["sizes"]
FIXTURES = list(C16.cfg["groups"]["F"])


def _grown(m, r):
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1,) * 2, np.uint8)) > 0


def _dist2(m, point):
    ys, xs = np.nonzero(m)
    return float(np.min((ys - point[0]) ** 2 + (xs - point[1]) ** 2)) if len(ys) else np.inf


def nearest_part(m, point):
    """The 8-connected part of mask m nearest to point (y, x)."""
    n, comp = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    parts = [comp == i for i in range(1, n)]
    return min(parts, key=lambda p: _dist2(p, point)) if parts else m


def wall_line(sk, hole, point=None):
    """(closed wall line inside the hole, horizontal?) -- the sketch's walls in and around the hole closed
    along their main direction, so gaps up to a door length are bridged. Without a point: the direction with
    more line pixels. With a point: the direction whose line passes nearest the point, and only the part of
    that line nearest the point (the wall segment at the flag)."""
    wall = (sk == I["wall"]) & _grown(hole, 3)
    if wall.sum() < 3:
        return None, None
    lines = {}
    for horiz in (True, False):
        L = SIZES["window"][1] + 4
        k = np.ones((1, L) if horiz else (L, 1), np.uint8)
        runs = cv2.morphologyEx(wall.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, 4) if horiz else (4, 1), np.uint8))
        closed = cv2.morphologyEx(runs, cv2.MORPH_CLOSE, k) > 0
        lines[horiz] = closed & hole
    if point is None:
        horiz = lines[True].sum() >= lines[False].sum()
    else:
        horiz = min((True, False), key=lambda h: (_dist2(lines[h], point), -lines[h].sum()))
    if not lines[horiz].any():
        return None, horiz
    return (lines[horiz] if point is None else nearest_part(lines[horiz], point)), horiz


def centre_segment(line, horiz, length, point=None):
    """The part of the line within `length` px (along it) of its middle, or of the line pixel nearest the point."""
    ys, xs = np.nonzero(line)
    along = xs if horiz else ys
    if point is None:
        mid = int(np.median(along))
    else:
        i = int(np.argmin((ys - point[0]) ** 2 + (xs - point[1]) ** 2))
        mid = int(along[i])
    keep = np.abs(along - mid) <= length // 2
    out = np.zeros_like(line)
    out[ys[keep], xs[keep]] = True
    return out


def place_opening(sk, hole, cls, exterior=False, point=None):
    line, horiz = wall_line(sk, hole, point)
    if line is None:
        return sk
    if exterior:
        out_side = _grown(np.isin(sk, C16.group("OUT")), 4)
        line = line & out_side
        if not line.any():
            return sk
    gap = line & (sk != I["wall"])
    if point is not None and gap.any():
        gap = nearest_part(gap, point)
    length = SIZES["door" if cls == I["door"] else "window"][1]
    seg = gap if 2 <= gap.sum() and cls == I["door"] else centre_segment(line, horiz, length, point)
    sk = sk.copy()
    sk[seg] = cls
    return sk


def place_fixture(sk, hole, name):
    a, b = SIZES[name]
    room = np.isin(sk, C16.group("S")) & hole
    wall = sk == I["wall"]
    touch = room & _grown(wall, 1)
    if not touch.any():
        return sk
    ys, xs = np.nonzero(touch)
    cy, cx = np.nonzero(hole)[0].mean(), np.nonzero(hole)[1].mean()
    i = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
    y, x = ys[i], xs[i]
    # which side the wall is on -> the box grows away from it
    dy = -1 if wall[max(0, y - 1), x] else (1 if wall[min(sk.shape[0] - 1, y + 1), x] else 0)
    dx = -1 if wall[y, max(0, x - 1)] else (1 if wall[y, min(sk.shape[1] - 1, x + 1)] else 0)
    h, w = (a, b) if dy != 0 else (b, a)                 # long side along the wall
    y0 = y if dy <= 0 else y - h + 1
    x0 = x if dx <= 0 else x - w + 1
    if dy == 0:
        y0 = y - h // 2
    if dx == 0:
        x0 = x - w // 2
    box = np.zeros_like(hole)
    box[max(0, y0):y0 + h, max(0, x0):x0 + w] = True
    sk = sk.copy()
    sk[box & np.isin(sk, C16.group("S")) & hole] = I[name]
    return sk


def close_wall(sk, hole, point=None, gap=None):
    sk = sk.copy()
    if gap is not None and gap.any():                    # the measured gap: painted as wall
        sk[gap] = I["wall"]
        return sk
    line, _ = wall_line(sk, hole, point)
    if line is None:
        return sk
    sk[line] = I["wall"]
    return sk


def apply_prior(sketch, hole, typ, rules=(), point=None, gap=None):
    """point: (y, x) in window px where the flag points (None: no location, e.g. C8); gap: bool mask (window
    px) of a measured wall gap to close (wall holes from the pipeline's gap rule)."""
    hole = hole.astype(bool)
    if typ == "door":
        return place_opening(sketch, hole, I["door"], point=point)
    if typ == "window":
        return place_opening(sketch, hole, I["window"], exterior=True, point=point)
    if typ in FIXTURES:
        return place_fixture(sketch, hole, typ)
    if typ == "wall":
        return close_wall(sketch, hole, point, gap)
    return sketch
