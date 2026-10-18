"""Realistic erasure of a component from a black-and-white DRAWING (user, overnight step 1).

    out = erase(img, mask)              # component removed, filled from its surroundings
    out = draw_wall(img, door, lbl)     # door kind (a) "opening forgotten": wall drawn across the doorway

erase       the component's own pixels (its rasterised SVG polygon = its label pixels), grown 2 px, are
            filled with cv2.inpaint (Telea, radius 3) from the pixels around them, so a room's grey or
            white fill continues through the spot instead of leaving a white patch. Never a box.
draw_wall   the doorway (the door's own label pixels) is painted in the local wall colour = median grey
            of the wall pixels within 3 wall thicknesses; the door polygon spans the wall, so the painted
            line has the local wall thickness. The door leaf and swing arc are drawn outside the door
            polygon in CubiCasa and are inpainted away too (dark pixels within one door length of the
            doorway on the room sides that are not wall in the label map).
Both return a single-channel uint8 image of the same size; figures draw their outlines on the axes only,
never into the image given to run_pipeline.
"""

import cv2
import numpy as np

from src.stage2.data.classes16 import C16

I = C16.index
GROW = 2                 # px the erased mask is grown by (antialiased edges)
RADIUS = 3               # cv2.inpaint radius


def grown(m, r):
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1,) * 2, np.uint8)) > 0


def erase(img, mask, lbl=None, grow=GROW, radius=RADIUS, wall_px=8):
    """Fill the component's pixels (grown `grow` px) from the surroundings. With the label map, the
    walls / openings / fixtures next to the spot are masked too while inpainting and put back after,
    so the fill comes from the room (or outside) side only and no wall colour smears into it."""
    m = grown(mask, grow)
    if not m.any():
        return img.copy()
    protect = np.zeros_like(m)
    if lbl is not None:
        solid = np.isin(lbl, C16.group("BOUNDARY") + C16.group("F")) & ~mask
        m &= ~solid                                    # the 2 px growth never eats into other walls
        protect = grown(m, 2 * wall_px) & solid
    out = cv2.inpaint(img, (m | protect).astype(np.uint8), radius, cv2.INPAINT_TELEA)
    out[protect] = img[protect]
    return out


def wall_colour(img, mask, lbl, wall_px=8):
    near = grown(mask, 3 * wall_px) & (lbl == I["wall"]) & ~mask
    return int(np.median(img[near])) if near.any() else 0


def bridge(door, lbl):
    """The wall continued across the doorway: wall mask closed along the wall direction, inside the
    door's box (grown 1 px), minus existing wall. -> (bridge mask, horizontal?)"""
    ys, xs = np.nonzero(door)
    horizontal = (xs.max() - xs.min()) >= (ys.max() - ys.min())
    L = int((xs.max() - xs.min()) if horizontal else (ys.max() - ys.min())) + 5
    k = np.ones((1, L) if horizontal else (L, 1), np.uint8)
    wall = (lbl == I["wall"]).astype(np.uint8)
    closed = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, k) > 0
    box = np.zeros(lbl.shape, bool)
    box[max(0, ys.min() - 1):ys.max() + 2, max(0, xs.min() - 1):xs.max() + 2] = True
    return closed & box & (wall == 0), horizontal


def door_symbol(img, door, lbl, ink=160, band=2):
    """Pixels of the door leaf + swing arc, found by geometry: the leaf is a line from a hinge corner of
    the doorway, perpendicular to the wall, one door width long; the arc is the quarter circle of that
    radius around the hinge. Of the 4 options (hinge at either end x swing to either side), the one with
    the most dark pixels in its bands wins. Only dark pixels in those bands (not wall / opening / fixture
    in the label map) are returned, so text and furniture nearby stay."""
    br, _ = bridge(door, lbl)
    ys, xs = np.nonzero(br if br.sum() >= 4 else door)  # the doorway proper (the polygon can be a box)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    horizontal = (x1 - x0) >= (y1 - y0)              # doorway runs along x (wall horizontal)
    L = float((x1 - x0) if horizontal else (y1 - y0)) + 1
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    keep = grown(np.isin(lbl, C16.group("BOUNDARY") + C16.group("F")) & ~door, 1)
    dark = (img < ink) & ~keep & ~door
    best, best_n = None, -1
    for hinge_end in (0, 1):
        for side in (-1, 1):
            if horizontal:                           # hinge on the wall face, leaf along y
                hx = x0 if hinge_end == 0 else x1
                hy = y0 if side < 0 else y1
                along, across = xx - hx, (yy - hy) * side
                toward = (along * (1 if hinge_end == 0 else -1))
            else:
                hy = y0 if hinge_end == 0 else y1
                hx = x0 if side < 0 else x1
                along, across = yy - hy, (xx - hx) * side
                toward = (along * (1 if hinge_end == 0 else -1))
            r = np.sqrt(along.astype(np.float32) ** 2 + across.astype(np.float32) ** 2)
            quadrant = (across >= -band) & (toward >= -band)
            arc = quadrant & (np.abs(r - L) <= band)
            leaf = (np.abs(toward) <= band) & (across >= 0) & (across <= L + band)
            m = (arc | leaf) & dark & ~np.isin(lbl, C16.group("OUT"))   # not across outdoor hatching
            if m.sum() > best_n:
                best, best_n = m, int(m.sum())
    return best | symbol_pieces(img, door, lbl, ink)


def symbol_pieces(img, door, lbl, ink=128, inside=0.8):
    """Connected dark pieces (not wall / opening / fixture) with >= 80% of their pixels within one door
    length of the doorway: catches thick arcs the thin geometric bands miss."""
    ys, xs = np.nonzero(door)
    length = int(max(ys.max() - ys.min(), xs.max() - xs.min())) + 1
    zone = grown(door, length)
    keep = grown(np.isin(lbl, C16.group("BOUNDARY") + C16.group("F")) & ~door, 1)
    dark = (img < ink) & ~keep & ~door
    y0, y1 = max(0, ys.min() - 3 * length), min(img.shape[0], ys.max() + 3 * length + 1)
    x0, x1 = max(0, xs.min() - 3 * length), min(img.shape[1], xs.max() + 3 * length + 1)
    n, comp = cv2.connectedComponents(dark[y0:y1, x0:x1].astype(np.uint8), connectivity=8)
    z = zone[y0:y1, x0:x1]
    out = np.zeros(img.shape, bool)
    for i in range(1, n):
        piece = comp == i
        if (piece & z).sum() >= inside * piece.sum():
            out[y0:y1, x0:x1] |= piece
    return out


def draw_wall(img, door, lbl, wall_px=8):
    """Door kind (a): the wall drawn across the doorway in the drawing's own style. The bridge (the wall
    continued across the gap) is painted line by line with the median grey of that line's wall pixels
    just past the doorway's ends (so outline-style walls get their two thin lines continued, solid walls
    their fill); the rest of the door polygon, the leaf and the arc are inpainted away. Without a bridge
    (no wall on either side) the door polygon is filled with the median local wall colour."""
    br, horizontal = bridge(door, lbl)
    out = erase(img, (door & ~br) | door_symbol(img, door, lbl), lbl, grow=1)
    if br.sum() < 4:
        out[door] = wall_colour(img, door, lbl, wall_px)
        return out
    wall = lbl == I["wall"]
    ys, xs = np.nonzero(br)
    near = grown(br, 4) & wall
    for v in np.unique(ys if horizontal else xs):  # one line of the bridge at a time
        line = (ys == v) if horizontal else (xs == v)
        src = near[v, :] if horizontal else near[:, v]
        vals = (img[v, :] if horizontal else img[:, v])[src]
        if len(vals):
            out[ys[line], xs[line]] = int(np.median(vals))
    return out


def open_gap(img, door, lbl, wall_px=8):
    """Door kind (b) "gap left, no door": doorway, leaf and arc erased; the wall ends stay."""
    return erase(img, door | door_symbol(img, door, lbl), lbl, wall_px=wall_px)
