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


def door_symbol(img, door, lbl, ink=128, inside=0.8):
    """Dark line pieces of the door leaf + swing arc: connected dark pieces (not wall / opening /
    fixture in the label map) with >= 80% of their pixels within one door length of the doorway.
    Pieces reaching further out (hatching, furniture, long dimension lines) are left alone."""
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
    """Door kind (a): leaf + arc inpainted away, doorway painted in the local wall colour."""
    out = erase(img, door_symbol(img, door, lbl), lbl, grow=1)
    out[door] = wall_colour(img, door, lbl, wall_px)
    return out


def open_gap(img, door, lbl, wall_px=8):
    """Door kind (b) "gap left, no door": doorway, leaf and arc erased; the wall ends stay."""
    return erase(img, door | door_symbol(img, door, lbl), lbl, wall_px=wall_px)
