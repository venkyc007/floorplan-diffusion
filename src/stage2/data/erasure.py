"""Realistic erasure of a component from a black-and-white DRAWING (user, overnight step 1; zones: 2026-10-19).

    out = erase(img, mask, lbl)          # these pixels removed, filled from their surroundings
    out = open_gap(img, door, lbl)       # door kind (b) "gap left, no door"
    out = draw_wall(img, door, lbl)      # door kind (a) "opening forgotten": wall drawn across the doorway
    out = erase_symbol(img, m, lbl)      # fixture (whole symbol) or window (symbol, then wall drawn across)

A symbol's ink reaches well beyond its CubiCasa polygon (a door's swing arc lies outside the door polygon;
shower hoses, sink outlines), so the WHOLE area the symbol can occupy is cleared, not just the polygon:
  door zone     the opening + a square of side = door length on BOTH sides of the wall (leaf and arc);
  symbol zone   fixture / window: the polygon's bounding box grown by 40% (20% of its size on each side).
Inside a zone, the component's pixels and every ink STROKE (>= 20 grey levels darker than the local paper /
room fill) are inpainted (cv2.inpaint, Telea, radius 3); straight runs longer than 60% (door zone) /
90% (symbol box) of the zone (hatching) stay. Other labelled components (walls, other doors / windows / railings / fixtures) and the swing-arc zones
of other doors are protected and put back unchanged. Text outside the zone is never touched (text inside it goes).
All functions return a single-channel uint8 image of the same size; figures draw their outlines on the axes
only, never into the image given to run_pipeline.
"""

import cv2
import numpy as np

from src.stage2.data.classes16 import C16

I = C16.index
GROW = 2                 # px the erased mask is grown by (antialiased edges)
RADIUS = 3               # cv2.inpaint radius
MARGIN = 3               # px added around a door zone (line width of the arc)
BOX_GROW = 0.2           # symbol box grown by 20% of its size on each side (40% in all)
INK = 20                 # a stroke is >= 20 grey levels darker than its local background
LONG = {"door": 0.6, "symbol": 0.9}   # straight runs longer than this share of the zone = background texture


def grown(m, r):
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1,) * 2, np.uint8)) > 0


def solid(lbl):
    return np.isin(lbl, C16.group("BOUNDARY") + C16.group("F"))


def erase(img, mask, lbl=None, grow=GROW, radius=RADIUS, wall_px=8, keep=None):
    """Fill `mask` (grown `grow` px) from the surroundings. With the label map, the walls / openings /
    fixtures next to the spot (and `keep`) are masked too while inpainting and put back after, so the
    fill comes from the room (or outside) side only and no wall colour smears into it."""
    m = grown(mask, grow)
    if not m.any():
        return img.copy()
    protect = np.zeros_like(m)
    if lbl is not None:
        hard = solid(lbl) & ~mask
        if keep is not None:
            hard |= keep
        m &= ~hard                                     # the growth never eats into other components
        protect = grown(m, 2 * wall_px) & hard
    out = cv2.inpaint(img, (m | protect).astype(np.uint8), radius, cv2.INPAINT_TELEA)
    out[protect] = img[protect]
    return out


def _box(m):
    ys, xs = np.nonzero(m)
    return ys.min(), ys.max(), xs.min(), xs.max()


def door_zone(door, shape, margin=MARGIN):
    """The opening + a square of side = door length on both sides of the wall (the leaf and the swing
    arc, whichever end the hinge is at and whichever side it opens to)."""
    y0, y1, x0, x1 = _box(door)
    z = np.zeros(shape, bool)
    if (x1 - x0) >= (y1 - y0):                         # opening runs along x: wall horizontal
        L = x1 - x0 + 1
        z[max(0, y0 - L - margin):y1 + L + margin + 1, max(0, x0 - margin):x1 + margin + 1] = True
    else:
        L = y1 - y0 + 1
        z[max(0, y0 - margin):y1 + margin + 1, max(0, x0 - L - margin):x1 + L + margin + 1] = True
    return z


def symbol_zone(m, shape, grow=BOX_GROW):
    y0, y1, x0, x1 = _box(m)
    dy, dx = int(np.ceil(grow * (y1 - y0 + 1))) + 1, int(np.ceil(grow * (x1 - x0 + 1))) + 1
    z = np.zeros(shape, bool)
    z[max(0, y0 - dy):y1 + dy + 1, max(0, x0 - dx):x1 + dx + 1] = True
    return z


def other_door_zones(lbl, but, min_px=20):
    """Swing-arc zones of every other door (kept when clearing a zone)."""
    n, comp = cv2.connectedComponents((lbl == I["door"]).astype(np.uint8), connectivity=4)
    z = np.zeros(lbl.shape, bool)
    for i in range(1, n):
        d = comp == i
        if d.sum() >= min_px and not (d & but).any():
            z |= door_zone(d, lbl.shape)
    return z


def strokes(img, zone, long_frac, ink=INK, bg_px=15):
    """Ink strokes inside the zone: pixels >= `ink` grey levels darker than the local paper / room fill
    (morphological closing removes thin dark lines), grown 1 px for antialiased edges, EXCEPT straight runs
    longer than `long_frac` of the zone (hatching, stripes of a grey fill: background texture). Door zone
    0.6: a leaf is at most ~50% of the zone across the wall, an arc has no long straight run. Symbol box
    0.9: a fixture's outline spans ~70% of its box (the box is grown 40%)."""
    bg = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((bg_px, bg_px), np.uint8))
    dark = (bg.astype(np.int16) - img > ink).astype(np.uint8)
    y0, y1, x0, x1 = _box(zone)
    long_ = (cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((1, int(long_frac * (x1 - x0 + 1))), np.uint8)) |
             cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((int(long_frac * (y1 - y0 + 1)), 1), np.uint8))) > 0
    return grown((dark > 0) & ~long_, 1) & zone


def clear_zone(img, comp, zone, lbl, long_frac, wall_px=8):
    """Inpaint the component's own pixels and every ink stroke in its zone (symbol lines, arc, leaf, text in
    the zone), keeping other components, other doors' arcs and background texture. Only strokes are filled,
    not the whole zone, so grey fills and hatching are not smeared."""
    keep = (solid(lbl) | other_door_zones(lbl, comp)) & ~comp     # the zone covers walls too: keep them
    return erase(img, comp | strokes(img, zone, long_frac), lbl, grow=1, wall_px=wall_px, keep=keep)


def wall_colour(img, mask, lbl, wall_px=8):
    near = grown(mask, 3 * wall_px) & (lbl == I["wall"]) & ~mask
    return int(np.median(img[near])) if near.any() else 0


def bridge(opening, lbl):
    """The wall continued across an opening: wall mask closed along the wall direction, inside the
    opening's box (grown 1 px), minus existing wall. -> (bridge mask, horizontal?)"""
    y0, y1, x0, x1 = _box(opening)
    horizontal = (x1 - x0) >= (y1 - y0)
    L = int((x1 - x0) if horizontal else (y1 - y0)) + 5
    k = np.ones((1, L) if horizontal else (L, 1), np.uint8)
    wall = (lbl == I["wall"]).astype(np.uint8)
    closed = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, k) > 0
    box = np.zeros(lbl.shape, bool)
    box[max(0, y0 - 1):y1 + 2, max(0, x0 - 1):x1 + 2] = True
    return closed & box & (wall == 0), horizontal


def paint_wall(out, img, opening, lbl, wall_px=8):
    """Draw the wall across `opening` in the drawing's own style: the bridge is painted line by line with
    the median grey of that line's wall pixels just past the opening's ends (outline-style walls get their
    two thin lines continued, solid walls their fill). No bridge (no wall on either side): the opening is
    filled with the median local wall colour."""
    br, horizontal = bridge(opening, lbl)
    if br.sum() < 4:
        out[opening] = wall_colour(img, opening, lbl, wall_px)
        return out
    ys, xs = np.nonzero(br)
    near = grown(br, 4) & (lbl == I["wall"])
    for v in np.unique(ys if horizontal else xs):      # one line of the bridge at a time
        line = (ys == v) if horizontal else (xs == v)
        src = near[v, :] if horizontal else near[:, v]
        vals = (img[v, :] if horizontal else img[:, v])[src]
        if len(vals):
            out[ys[line], xs[line]] = int(np.median(vals))
    return out


def open_gap(img, door, lbl, wall_px=8):
    """Door kind (b): opening, leaf and arc cleared (the door zone); the wall ends stay."""
    return clear_zone(img, door, door_zone(door, img.shape), lbl, LONG["door"], wall_px)


def draw_wall(img, door, lbl, wall_px=8):
    """Door kind (a): the door zone cleared as in (b), then the wall drawn across the doorway."""
    return paint_wall(open_gap(img, door, lbl, wall_px), img, door, lbl, wall_px)


def erase_symbol(img, m, lbl, wall_px=8):
    """Fixture: its symbol box (grown 40%) cleared. Window: the same, then the wall drawn across it
    (an omitted window = a plain wall in the drawing)."""
    out = clear_zone(img, m, symbol_zone(m, img.shape), lbl, LONG["symbol"], wall_px)
    if np.bincount(lbl[m], minlength=C16.n).argmax() == I["window"]:
        out = paint_wall(out, img, m, lbl, wall_px)
    return out
