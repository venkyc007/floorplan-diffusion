"""Large image-level omissions (user, 2026-10-20; deliverable 5 needs a size breakdown).

    o = erase_room(img, lbl, rng)      # "room (whole)": a room merged into its neighbours
    o = erase_region(img, lbl, rng)    # "region": everything inside a rectangle except the outer building walls
    -> (erased drawing, structure mask, area mask) or None

Both work on the parser-scale drawing + its label map (walls ~8 px) and reuse the zone erasure of
src/stage2/data/erasure.py, so the result looks like a drawing that never had those parts:
  room (whole)  a bedroom / bathroom / kitchen / other_room instance of 2-15% of the building. Erased: its
                interior walls (wall pixels within 2 wall thicknesses of the room and NOT within 2 wall
                thicknesses of outdoor / background, i.e. the outer building walls stay), the doors in them
                (zone cleared, leaf and arc too), its windows (wall drawn across, as for a window omission),
                exterior doors (wall drawn across) and its fixtures; then every ink stroke left inside the merged room
                (symbol remains, smears, its name text: a merged room has no label of its own) is filled with the
                local paper grey, as for the region.
  region        a rectangle (aspect 0.6-1.6) covering 5-25% of the building; erased: every wall, door, window,
                railing and fixture inside it except the outer building walls (openings in those are closed with
                wall), then every remaining ink stroke inside (text, stairs, hatching ends), filled with the local
                paper / room grey. Needs >= 100 px of interior walls / doors inside, else retried.
structure mask = the erased components' own pixels (walls, doors, windows, fixtures, railings): what "found" and
"fixed" are scored on. area mask = the room + its erased walls / the rectangle inside the building: what the
size bin is measured on (omission_meta).
"""

import numpy as np

from src.stage2.constraints.rules import components, room_instances
from src.stage2.data.classes16 import C16
from src.stage2.data.erasure import LONG, clear_zone, draw_wall, erase, erase_symbol, grown, open_gap

I = C16.index
WALL_PX = 8
ROOM_TYPES = ["bedroom", "bathroom", "kitchen", "other_room"]
ROOM_PCT = (2.0, 15.0)
REGION_PCT = (5.0, 25.0)
TRIES = 30


def _outer(lbl, wall_px=WALL_PX):
    """Pixels within 2 wall thicknesses of outdoor / background: the outer building walls live there."""
    return grown(np.isin(lbl, C16.group("OUT")), 2 * wall_px)


def _pct(m, lbl):
    return 100.0 * m.sum() / max(int((~np.isin(lbl, C16.group("OUT"))).sum()), 1)


def _erase_parts(img, lbl, walls, doors, windows, symbols, outer, filler, wall_px=WALL_PX):
    """Erase in a fixed order on a working label map (erased parts become `filler`, so later steps do not
    protect them): windows and exterior doors closed with wall, interior doors cleared, fixtures / railings
    cleared, then the interior walls (and the interior doorways) filled from the rooms."""
    out, lw = img.copy(), lbl.copy()
    for w in windows:
        out = erase_symbol(out, w, lw, wall_px)
        lw[w] = I["wall"]                               # now drawn as wall: part of the outer wall
    inner_doors = []
    for d in doors:
        if (grown(d, 2 * wall_px) & np.isin(lbl, C16.group("OUT"))).any():
            out = draw_wall(out, d, lw, wall_px)        # exterior door: the outer wall closed
            lw[d] = I["wall"]
        else:
            out = open_gap(out, d, lw, wall_px)
            inner_doors.append(d)
    for s in symbols:
        out = erase_symbol(out, s, lw, wall_px)
        lw[s] = filler
    gone = walls.copy()
    for d in inner_doors:
        gone |= d
    if gone.any():
        out = erase(out, gone, lw, wall_px=wall_px)
        lw[gone] = filler
    return out, lw


def _clear_rest(out, lw, zone, wall_px=WALL_PX):
    """Every ink stroke left in the zone (symbol remains, inpainting smears, text) filled with the local paper /
    room grey; the walls still in the working label (outer walls, closed openings) and other components are kept."""
    zone = zone & ~grown(lw == I["wall"], 1)
    return clear_zone(out, np.zeros(lw.shape, bool), zone, lw, LONG["symbol"], median=True, wall_px=wall_px)


def erase_room(img, lbl, rng, pct=ROOM_PCT, wall_px=WALL_PX):
    rooms = [(c, m) for c, m in room_instances(lbl) if C16.names[c] in ROOM_TYPES and pct[0] <= _pct(m, lbl) <= pct[1]]
    if not rooms:
        return None
    c, room = rooms[rng.integers(len(rooms))]
    near = grown(room, 2 * wall_px)
    outer = _outer(lbl, wall_px)
    walls = near & (lbl == I["wall"]) & ~outer
    doors = [d for d in components(lbl == I["door"]) if (d & grown(room, 3)).any()]      # in the room's walls
    windows = [w for w in components(lbl == I["window"]) if (w & near).sum() >= 0.5 * w.sum()]
    symbols = [f for f in components(np.isin(lbl, C16.group("F"))) if (f & grown(room, 2)).sum() >= 0.5 * f.sum()]
    if walls.sum() < 50:
        return None
    out, lw = _erase_parts(img, lbl, walls, doors, windows, symbols, outer, c, wall_px)
    out = _clear_rest(out, lw, grown(room | walls, 2), wall_px)
    structure = walls.copy()
    for m in doors + windows + symbols:
        structure |= m
    return out, structure, room | structure


def erase_region(img, lbl, rng, pct=REGION_PCT, wall_px=WALL_PX):
    H, W = lbl.shape
    building = ~np.isin(lbl, C16.group("OUT"))
    by, bx = np.nonzero(building)
    if len(by) == 0:
        return None
    outer = _outer(lbl, wall_px)
    for _ in range(TRIES):
        target = rng.uniform(*pct) / 100 * building.sum()
        aspect = rng.uniform(0.6, 1.6)
        h, w = int(np.sqrt(target * aspect)), int(np.sqrt(target / aspect))
        k = rng.integers(len(by))
        y0, x0 = max(0, by[k] - h // 2), max(0, bx[k] - w // 2)
        rect = np.zeros(lbl.shape, bool)
        rect[y0:min(H, y0 + h), x0:min(W, x0 + w)] = True
        area = rect & building
        if not pct[0] <= _pct(area, lbl) <= pct[1]:
            continue
        inside = lambda m: (m & rect).sum() >= 0.5 * m.sum()
        walls = rect & (lbl == I["wall"]) & ~outer
        doors = [d for d in components(lbl == I["door"]) if inside(d)]
        if walls.sum() + sum(int(d.sum()) for d in doors if not (d & outer).any()) < 100:
            continue
        windows = [m for m in components(lbl == I["window"]) if inside(m)]
        symbols = [m for m in components(np.isin(lbl, C16.group("F") + [I["railing"]])) if inside(m)]
        out, lw = _erase_parts(img, lbl, walls, doors, windows, symbols, outer, I["other_room"], wall_px)
        out = _clear_rest(out, lw, rect, wall_px)
        structure = walls.copy()
        for m in doors + windows + symbols:
            structure |= m
        return out, structure, area
    return None
