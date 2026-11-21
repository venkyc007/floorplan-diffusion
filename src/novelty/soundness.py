"""Structural soundness of a completed plan: the three checks the instructor named, with no ground truth needed.

    report = soundness(plan, region)            # plan = 16-class label map, region = bool mask (the omission's box)

Why this file: the project's rule check (src/stage2/pipeline/e2e_val.rule_violations) counts violation blobs of
18 soft constraint maps in a window, and its "valid" compares that count with the clean TRUE drawing. On real
drawings most of that count comes from the parser's map around the hole: pasting the true layout into the erased
area of the parser's map scores MORE violations than the fills do (29.7 vs 20-24 per large case, 30 tuning cases).
So it cannot say whether a fill is sound. These checks look only at the elements in the omission's region,
one by one, and ask the three questions of the review:

  1. walls connect      dead_ends  wall ends inside the region that touch no other wall, door, window, railing or
                                   the outside (a wall stopping in the middle of a room)
                        floating   groups of walls / openings lying wholly inside the region, touching nothing
                                   outside it (a piece hanging in the air)
  2. openings in walls  bad_openings  doors / windows whose two ends do not both meet a wall (or another opening)
                                   (not judged: which side a window faces - CubiCasa has glazed inner walls)
  3. rooms closed       leaks      places where a room touches the outside (outdoor / background) with no wall,
                                   door, window or railing between
                        doorless   rooms (spaces closed by walls) in the region that no door reaches
  also reported         mixed      places where two different room types meet with no wall between. Real plans
                                   have open-plan kitchens and hallways, so this is reported but not part of "sound".

A plan is SOUND in the region when 1, 2 and 3 all pass (every count above except `mixed` is 0). Real drawings do
not always pass (a free-standing wall, a door the drawing leaves out), so every table also gives the true layout's
own rate. How it fits in: used by src/novelty/eval_e2e.py (columns s_*), by the repair step
(src/novelty/repair.py) to accept or reject a change, and by the figures (src/novelty/assessment_figures.py).
Pure geometry, no model.
"""

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from src.stage2.data.classes16 import C16

WALL, DOOR, WINDOW, RAIL = (C16.index[n] for n in ("wall", "door", "window", "railing"))
BOUNDARY = C16.group("BOUNDARY")
ROOMS = C16.group("S")
OUT = C16.group("OUT")
FIX = C16.group("F")
SOLID = [WALL, DOOR, WINDOW]   # railings stand free inside rooms (stairs), so they are not judged as floating
K8 = np.ones((3, 3), bool)
K4 = ndimage.generate_binary_structure(2, 1)
MIN_ROOM = 30          # px: smaller closed spaces are gaps inside a wall drawing, not rooms
SPUR = 6               # px: a skeleton branch shorter than this ending at a junction is a corner artefact
REACH = 2              # px: how near a wall end must be to something to count as meeting it


def _grow(m, r):
    return ndimage.binary_dilation(m, K8, iterations=r) if r > 0 else m


def _crop(region, margin, shape):
    ys, xs = np.nonzero(region)
    if not len(ys):
        return None
    return (slice(max(0, ys.min() - margin), min(shape[0], ys.max() + margin + 1)),
            slice(max(0, xs.min() - margin), min(shape[1], xs.max() + margin + 1)))


def wall_ends(plan, region):
    """[(y, x)] dead ends of walls inside `region` (whole-plan coordinates) and the skeleton they came from."""
    sl = _crop(region, 24, plan.shape)
    if sl is None:
        return [], None
    p, reg = plan[sl], region[sl]
    wall = p == WALL
    sk = skeletonize(ndimage.binary_closing(wall, K8) & _grow(wall, 1) | wall)
    nb = ndimage.convolve(sk.astype(np.uint8), K8.astype(np.uint8), mode="constant") - sk
    ends = np.argwhere(sk & (nb == 1) & reg)
    other = np.isin(p, [DOOR, WINDOW, RAIL] + OUT + FIX)        # a wall may also end at a counter / fixture
    H, W = p.shape
    out = []
    for y, x in ends:
        if (y + sl[0].start in (0, plan.shape[0] - 1)) or (x + sl[1].start in (0, plan.shape[1] - 1)):
            continue                                      # the map's own edge
        if _is_spur(sk, nb, y, x):
            continue
        # the wall pixels this end belongs to: its tip region; does anything other than its own wall touch it?
        y0, y1, x0, x1 = max(0, y - 4), min(H, y + 5), max(0, x - 4), min(W, x + 5)
        if other[y0:y1, x0:x1].any():
            continue
        if _faces_wall(wall, sk, y, x):
            continue
        out.append((int(y + sl[0].start), int(x + sl[1].start)))
    return out, sk


def _is_spur(sk, nb, y, x):
    """True if the branch from this end reaches a junction within SPUR px (a corner / thickness artefact)."""
    seen, cur = {(y, x)}, (y, x)
    for _ in range(SPUR):
        cy, cx = cur
        nxt = [(cy + dy, cx + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx)
               and 0 <= cy + dy < sk.shape[0] and 0 <= cx + dx < sk.shape[1] and sk[cy + dy, cx + dx]
               and (cy + dy, cx + dx) not in seen]
        if not nxt:
            return True                                   # a tiny isolated stub: counted by `floating`, not here
        if len(nxt) > 1 or nb[nxt[0]] >= 3:
            return True
        cur = nxt[0]
        seen.add(cur)
    return False


def _faces_wall(wall, sk, y, x):
    """True if, straight ahead of the end (away from its own branch), another wall begins within REACH + 1 px of
    the end of this wall's pixels: the wall meets it (drawn with a hairline gap)."""
    ys, xs = np.nonzero(sk[max(0, y - 3):y + 4, max(0, x - 3):x + 4])
    dy, dx = y - (ys.mean() + max(0, y - 3)), x - (xs.mean() + max(0, x - 3))
    if abs(dy) < 1e-6 and abs(dx) < 1e-6:
        return False
    if abs(dx) >= abs(dy):
        sx, sy = int(np.sign(dx)), 0
    else:
        sx, sy = 0, int(np.sign(dy))
    lab, _ = ndimage.label(wall, K8)
    own = lab[y, x]
    cy, cx = y, x
    while 0 <= cy < wall.shape[0] and 0 <= cx < wall.shape[1] and wall[cy, cx]:
        cy, cx = cy + sy, cx + sx                         # leave this wall's own pixels
    for k in range(REACH + 1):
        ty, tx = cy + k * sy, cx + k * sx
        if 0 <= ty < wall.shape[0] and 0 <= tx < wall.shape[1] and wall[ty, tx] and lab[ty, tx] != own:
            return True
    return False


def floating(plan, region):
    """Groups of wall / door / window pixels (8-connected) of at least 6 px lying wholly inside
    the region: they touch nothing outside it."""
    b = np.isin(plan, SOLID)
    lab, n = ndimage.label(b, K8)
    if not n:
        return []
    inside = ndimage.sum(region, lab, range(1, n + 1))
    size = ndimage.sum(b, lab, range(1, n + 1))
    return [k + 1 for k in range(n) if size[k] >= 6 and inside[k] == size[k]]


def openings_status(plan, region):
    """[(component mask slice, mask, ok)] for every door / window (>= 4 px) touching the region. ok = both ends
    meet a wall or another boundary element; the ends are the 3 px strips just beyond the opening along its long
    side."""
    bnd = np.isin(plan, BOUNDARY)
    out = []
    for cls in (DOOR, WINDOW):
        lab, n = ndimage.label(plan == cls, K8)
        for k, sl in enumerate(ndimage.find_objects(lab), 1):
            comp = lab[sl] == k
            if comp.sum() < 4 or not region[sl][comp].any():
                continue
            h, w = comp.shape
            (y0, y1), (x0, x1) = (sl[0].start, sl[0].stop), (sl[1].start, sl[1].stop)
            other = bnd & (lab != k)
            if w >= h:                                    # horizontal: look left and right
                a, b = other[y0:y1, max(0, x0 - 3):x0], other[y0:y1, x1:x1 + 3]
            else:
                a, b = other[max(0, y0 - 3):y0, x0:x1], other[y1:y1 + 3, x0:x1]
            ok = bool(a.any() and b.any())
            out.append((sl, comp, ok))
    return out


def bad_openings(plan, region):
    """(bad, total) doors / windows in the region that do not sit in a wall line."""
    st = openings_status(plan, region)
    return sum(not ok for *_, ok in st), len(st)


def _contacts(a, b, region, min_px=2):
    """Connected places (>= min_px px) inside the region where mask a touches mask b (4-neighbours)."""
    touch = a & ndimage.binary_dilation(b, K4) & region
    lab, n = ndimage.label(touch, K8)
    if not n:
        return 0
    return int((np.bincount(lab.ravel())[1:] >= min_px).sum())


def leaks(plan, region):
    return _contacts(np.isin(plan, ROOMS), np.isin(plan, OUT), region)


def mixed(plan, region):
    n = 0
    rooms = [c for c in ROOMS if (plan[region] == c).any()]
    for i, a in enumerate(rooms):
        for b in rooms[i + 1:]:
            n += _contacts(plan == a, plan == b, region, 3)
    return n


def doorless(plan, region):
    """Rooms (connected non-boundary areas, mostly room types, >= MIN_ROOM px) touching the region that no door
    reaches. Areas touching the map's edge or the outside are not judged (they are not closed rooms)."""
    space = ~np.isin(plan, BOUNDARY)
    lab, n = ndimage.label(space, K4)
    door = ndimage.binary_dilation(plan == DOOR, K8, iterations=2)
    edge = np.zeros_like(space)
    edge[0], edge[-1], edge[:, 0], edge[:, -1] = True, True, True, True
    out = 0
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        comp = lab[sl] == k
        if comp.sum() < MIN_ROOM or not region[sl][comp].any() or edge[sl][comp].any():
            continue
        vals = plan[sl][comp]
        if np.isin(vals, OUT).mean() > 0.2 or np.isin(vals, ROOMS).mean() < 0.5:
            continue
        out += not door[sl][comp].any()
    return out


def soundness(plan, region):
    """All checks in `region` -> dict of counts + the three pass / fail flags + `sound`."""
    region = region.astype(bool)
    ends, _ = wall_ends(plan, region)
    fl = floating(plan, region)
    bad, total = bad_openings(plan, region)
    r = {"dead_ends": len(ends), "floating": len(fl), "bad_openings": bad, "openings": total,
         "leaks": leaks(plan, region), "doorless": doorless(plan, region), "mixed": mixed(plan, region)}
    r["walls_connect"] = r["dead_ends"] == 0 and r["floating"] == 0
    r["openings_in_walls"] = r["bad_openings"] == 0
    r["rooms_closed"] = r["leaks"] == 0 and r["doorless"] == 0
    r["sound"] = r["walls_connect"] and r["openings_in_walls"] and r["rooms_closed"]
    return r


def problem_map(plan, region):
    """{check: bool mask} of where each problem is, for figures (ends and openings as small squares)."""
    region = region.astype(bool)
    H, W = plan.shape
    m = {k: np.zeros((H, W), bool) for k in ("dead_ends", "floating", "bad_openings", "leaks", "doorless", "mixed")}
    for y, x in wall_ends(plan, region)[0]:
        m["dead_ends"][max(0, y - 3):y + 4, max(0, x - 3):x + 4] = True
    lab, _ = ndimage.label(np.isin(plan, SOLID), K8)
    for k in floating(plan, region):
        m["floating"] |= lab == k
    for sl, comp, ok in openings_status(plan, region):
        if not ok:
            m["bad_openings"][sl] |= comp
    m["bad_openings"] = _grow(m["bad_openings"], 1)
    bnd = np.isin(plan, BOUNDARY)
    rooms, outs = np.isin(plan, ROOMS), np.isin(plan, OUT)
    m["leaks"] = _grow(rooms & ndimage.binary_dilation(outs, K4) & region, 1)
    for i, a in enumerate(ROOMS):
        for b2 in ROOMS[i + 1:]:
            m["mixed"] |= (plan == a) & ndimage.binary_dilation(plan == b2, K4) & region
    space = ~bnd
    lab, _ = ndimage.label(space, K4)
    door = ndimage.binary_dilation(plan == DOOR, K8, iterations=2)
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        comp = lab[sl] == k
        if comp.sum() < MIN_ROOM or not region[sl][comp].any():
            continue
        if sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == H or sl[1].stop == W:
            continue
        vals = plan[sl][comp]
        if np.isin(vals, OUT).mean() > 0.2 or np.isin(vals, ROOMS).mean() < 0.5 or door[sl][comp].any():
            continue
        m["doorless"][sl] |= comp
    return m
