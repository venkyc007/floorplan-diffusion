"""Door-ink check (user, 2026-10-19): a parsed door with no door symbol (leaf / swing arc) drawn next to it =
candidate "door symbol missing" (door kind (b): the opening was left but the door was not drawn).

    scores = door_ink_scores(image, label)          # [(door mask in label px, score)]
    cands = ink_candidates(image, label, threshold)  # Candidate("door", box, ["INK"], score) for score < threshold

For each parsed door (4-connected door component of the completion-scale label, >= 6 px) the door's box is
mapped to the drawing; its swing zone = the door zone of src/stage2/data/erasure.py (opening + a square of side
= door length on BOTH sides of the wall). Ink there = stroke pixels (>= 20 grey levels darker than the local
background, straight runs longer than 60% of the zone are texture), minus the parsed walls / windows / railings /
fixtures (grown 1 px) and the opening itself. Score = ink pixels / door length (px): an arc + leaf drawn with
~1-2 px lines gives ~2.5-5; a bare opening ~0. Computed in a local window (fast on big drawings).
Guard: no candidates in a drawing where more than half of the parsed doors score below the threshold.
Restriction (user, 2026-10-19): candidates only at doors where a missing door matters = R5's pairs (relevant()).
"""

import cv2
import numpy as np

from src.stage2.constraints.rules import components
from src.stage2.data.classes16 import C16
from src.stage2.data.erasure import INK, LONG, door_zone
from src.stage2.constraints.constraints import CONST
from src.stage2.detect import rules as R
from src.stage2.detect.gaps import closed_pairs, instance_areas, needs_door

I = C16.index
SOLID = [I["wall"], I["window"], I["railing"]] + C16.group("F")


def relevant(label, d, gap, bad=None, area=None):
    """Is this a door where a missing door matters (user, 2026-10-19)? The same pairs R5 uses: both faces are
    rooms (exterior doors skipped), the pair is a closed pair (direct-contact rate < open_rate; open-plan pairs
    skipped), and one side is a bedroom / bathroom / other_room >= need_area while the other is not a closet."""
    bad = closed_pairs(open_rate=gap.get("open_rate", 0.10)) if bad is None else bad
    area = instance_areas(label) if area is None else area
    ys, xs = np.nonzero(d)
    h, w = label.shape
    k = CONST["T_wall"] + 2
    horizontal = (xs.max() - xs.min()) >= (ys.max() - ys.min())
    faces = []
    for sgn in (-1, 1):
        fy = np.clip(ys + sgn * k, 0, h - 1) if horizontal else ys
        fx = xs if horizontal else np.clip(xs + sgn * k, 0, w - 1)
        faces.append((fy, fx))
    sides = [int(np.bincount(label[fy, fx]).argmax()) for fy, fx in faces]
    if not all(c in C16.group("S") for c in sides) or sides[0] == sides[1] or not bad[sides[0], sides[1]]:
        return False
    return needs_door(label, area, *faces[0], *faces[1], gap.get("need_area", 0), gap.get("small_other", 0))


def door_ink_scores(image, label, ink=INK, min_px=6, gap=None):
    """[(door mask, score)]; with `gap` (the R5 gap_rule settings) only the relevant doors (see relevant)."""
    H, W = image.shape
    h, w = label.shape
    fy, fx = H / h, W / w
    out = []
    bad = closed_pairs(open_rate=gap.get("open_rate", 0.10)) if gap is not None else None
    area = instance_areas(label) if gap is not None else None
    for d in components(label == I["door"]):
        if d.sum() < min_px:
            continue
        if gap is not None and not relevant(label, d, gap, bad, area):
            continue
        ys, xs = np.nonzero(d)
        y0, y1 = int(ys.min() * fy), int((ys.max() + 1) * fy)
        x0, x1 = int(xs.min() * fx), int((xs.max() + 1) * fx)
        L = max(y1 - y0, x1 - x0)
        m = 2 * L + 20                                    # local window around the door zone
        a0, a1, b0, b1 = max(0, y0 - m), min(H, y1 + m), max(0, x0 - m), min(W, x1 + m)
        img = image[a0:a1, b0:b1]
        door = np.zeros(img.shape, bool)
        door[y0 - a0:y1 - a0, x0 - b0:x1 - b0] = True
        zone = door_zone(door, img.shape)
        bg = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        dark = ((bg.astype(np.int16) - img) > ink).astype(np.uint8)
        zy, zx = np.nonzero(zone)
        lh, lv = max(2, int(LONG["door"] * (zx.max() - zx.min() + 1))), max(2, int(LONG["door"] * (zy.max() - zy.min() + 1)))
        long_ = (cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((1, lh), np.uint8)) |
                 cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((lv, 1), np.uint8))) > 0
        lab = cv2.resize(label[int(a0 / fy):int(np.ceil(a1 / fy)), int(b0 / fx):int(np.ceil(b1 / fx))], None,
                         fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
        solid = np.zeros(img.shape, bool)
        hh, ww = min(lab.shape[0], img.shape[0]), min(lab.shape[1], img.shape[1])
        solid[:hh, :ww] = np.isin(lab[:hh, :ww], SOLID)
        solid = cv2.dilate(solid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        strokes = (dark > 0) & ~long_ & zone & ~solid & ~door
        out.append((d, float(strokes.sum()) / max(L, 1)))
    return out


GUARD = 0.5     # user (2026-10-19): skip the check when more than half of a drawing's parsed doors have low ink


def guarded(scores, threshold, guard=GUARD):
    """Per-drawing guard: a drawing whose doors are mostly drawn without a symbol (sliding doors, simplified
    styles) gets no ink candidates at all."""
    low = [s < threshold for s in scores]
    return bool(low) and float(np.mean(low)) > guard


def ink_candidates(image, label, threshold, guard=GUARD, gap=None):
    """Guard on ALL parsed doors (a drawing style without arcs); candidates only among the relevant doors."""
    scores = door_ink_scores(image, label)
    if guarded([s for _, s in scores], threshold, guard):
        return []
    if gap is not None:
        bad, area = closed_pairs(open_rate=gap.get("open_rate", 0.10)), instance_areas(label)
        scores = [(d, s) for d, s in scores if relevant(label, d, gap, bad, area)]
    return [R.Candidate("door", R.box_of(d, label.shape), ["INK"], s) for d, s in scores if s < threshold]
