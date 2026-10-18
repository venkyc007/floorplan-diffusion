"""Simulated drafting omissions for the missing-component detector (spec section 6).

Given a ground-truth 16-class plan (completion scale), remove 1-3 components the way a
draftsperson would forget them, and record WHERE and WHAT is missing:

    omitted        the plan afterwards
    door           closed with wall (50%) or a plain gap of the neighbouring space (50%)
    window         wall
    wall segment   the neighbouring space extends through it
    railing        outdoor / background (whatever lies next to it)
    fixture        the surrounding room type
    room           its interior walls/doors removed and its type merged into a neighbour

Target = per-pixel missing type: 0 none, 1 wall, 2 door, 3 window, 4 railing, 5 room,
6 toilet, 7 sink, 8 bath_shower, 9 appliance (MISSING list below).
20% of samples stay intact (all "none"), to teach the model not to raise false alarms.

Parser-like noise (NOT labelled as missing): a few tiny wall breaks (2-4 px, filled by the
neighbouring space) and 1-2% of pixels flipped to the class of a random nearby pixel.
"""

import cv2
import numpy as np

from src.stage2.complete.windows import _wall_piece
from src.stage2.data.classes16 import C16

I = C16.index
MISSING = ["none", "wall", "door", "window", "railing", "room", "toilet", "sink", "bath_shower", "appliance"]
M = {n: i for i, n in enumerate(MISSING)}
KINDS = ["door", "window", "wall", "railing", "fixture", "room"]
SPACES = C16.group("S") + C16.group("OD") + C16.group("BG")
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)


def components(mask):
    n, comp = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    return [comp == i for i in range(1, n)]


def grow_fill(label, region, allowed, max_steps=60):
    """Fill `region` pixels with the classes in `allowed` that grow into it from its border
    (one 4-neighbour step at a time; the most common grower wins per step). Returns a copy."""
    out = label.copy()
    todo = region.copy()
    for _ in range(max_steps):
        if not todo.any():
            break
        changed = False
        for c in allowed:
            grow = (cv2.dilate((out == c).astype(np.uint8), CROSS) > 0) & todo
            if grow.any():
                out[grow] = c
                todo &= ~grow
                changed = True
        if not changed:
            break
    return out


def omit(label, kind, rng):
    """(new label, missing mask, missing type index) or None if the plan has nothing of that kind."""
    if kind in ("door", "window", "railing"):
        comps = components(label == I[kind])
        if not comps:
            return None
        m = comps[rng.integers(len(comps))]
        out = label.copy()
        if kind == "window" or (kind == "door" and rng.random() < 0.5):
            out[m] = I["wall"]
        else:
            allowed = SPACES if kind == "door" else C16.group("OD") + C16.group("BG")
            out = grow_fill(label, m, allowed)
        return out, m, M[kind]
    if kind == "fixture":
        comps = components(np.isin(label, C16.group("F")))
        if not comps:
            return None
        m = comps[rng.integers(len(comps))]
        cls = int(np.bincount(label[m]).argmax())
        return grow_fill(label, m, C16.group("S")), m, M[C16.names[cls]]
    if kind == "wall":
        m = _wall_piece(label, rng)
        if m is None:
            return None
        return grow_fill(label, m, SPACES), m, M["wall"]
    if kind == "room":
        rooms = [m for m in components(np.isin(label, C16.group("S"))) if 20 <= m.sum()]
        rng.shuffle(rooms)
        for m in rooms:
            ring = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            boundary = ring & np.isin(label, [I["wall"], I["door"]])
            neigh = label[ring & ~m & np.isin(label, C16.group("S"))]
            if len(neigh) == 0 or not boundary.any():
                continue
            other = int(np.bincount(neigh).argmax())
            # interior walls/doors between this room and the neighbour: boundary pixels next to both
            near_other = cv2.dilate((label == other).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            walls = boundary & near_other & (cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
            out = label.copy()
            out[m | walls] = other
            return out, m | walls, M["room"]
        return None
    raise KeyError(kind)


def parser_noise(label, rng, breaks=(0, 3), flip=(0.01, 0.02)):
    """Tiny wall breaks + random class flips (look like parser errors; not labelled missing)."""
    out = label.copy()
    walls = np.argwhere(out == I["wall"])
    for _ in range(int(rng.integers(breaks[0], breaks[1] + 1))):
        if len(walls) == 0:
            break
        y, x = walls[rng.integers(len(walls))]
        s = int(rng.integers(2, 5))
        m = np.zeros(out.shape, bool)
        m[y:y + s, x:x + s] = True
        out = grow_fill(out, m & (out == I["wall"]), SPACES, max_steps=4)
    h, w = out.shape
    n = int(rng.uniform(*flip) * h * w)
    ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
    ny = np.clip(ys + rng.integers(-2, 3, n), 0, h - 1)
    nx = np.clip(xs + rng.integers(-2, 3, n), 0, w - 1)
    out[ys, xs] = out[ny, nx]
    return out


def simulate(label, rng, intact_p=0.2, n_omit=(1, 3)):
    """(plan with omissions + noise, target (H, W) uint8 of MISSING indices, list of omitted kinds)."""
    target = np.zeros(label.shape, np.uint8)
    out, kinds = label.copy(), []
    if rng.random() >= intact_p:
        for _ in range(int(rng.integers(n_omit[0], n_omit[1] + 1))):
            for kind in rng.permutation(KINDS):
                r = omit(out, kind, rng)
                if r is not None:
                    out, m, t = r
                    target[m & (target == 0)] = t
                    kinds.append(str(kind))
                    break
    return parser_noise(out, rng), target, kinds


# ---------------------------------------------------------------- one omission with metadata

SIZE_BINS = (0.005, 0.02)      # omitted area / building area: small < 0.5% <= medium < 2% <= large


def building_area(label):
    return int(np.isin(label, C16.group("BOUNDARY") + C16.group("S") + C16.group("F")).sum())


def describe(label, m):
    """Metadata of an omitted region m on the ORIGINAL plan: location (exterior if it touches the
    outside) and size (share of the building area, binned)."""
    touches_out = (cv2.dilate(m.astype(np.uint8), CROSS) > 0) & np.isin(label, C16.group("OUT"))
    share = m.sum() / max(building_area(label), 1)
    size = "small" if share < SIZE_BINS[0] else ("medium" if share < SIZE_BINS[1] else "large")
    return {"location": "exterior" if touches_out.any() else "interior", "size_bin": size,
            "size_pct": round(100 * float(share), 3)}


def simulate_one(label, rng, kind=None, noise=False):
    """Exactly one omission (of `kind`, or a random available kind) ->
    dict(plan, mask, kind, type (MISSING name), location, size_bin, size_pct), or None."""
    for k in ([kind] if kind else list(rng.permutation(KINDS))):
        r = omit(label, k, rng)
        if r is None:
            continue
        plan, m, t = r
        if noise:
            plan = parser_noise(plan, rng)
        return {"plan": plan, "mask": m, "kind": str(k), "type": MISSING[t], **describe(label, m)}
    return None
