"""Novelty round 4 - SOUNDNESS: repair a region fill so that it passes the three structural checks.

    new_map, info = repair_fill(before, after, hole)        # after = the fill (best after the round-3 clean-up)

Why: after the round-3 clean-up the fills look drawn by rule (straight walls, one thickness, no blobs), but the
structural checks of src/novelty/soundness.py still fail on most large fills: the fill splits space into rooms that
no door reaches, leaves bits of "outside" inside the building, and puts doors / windows where no wall holds them.
The true layout passes those checks in about 9 cases out of 10. Each repair below fixes one kind of problem, and a
change is KEPT only if the soundness problems in the hole do not increase and the one it targets goes down
(so a repair can never make the plan less sound by these checks):

  floating    a wall group the fill drew that touches nothing outside the hole is joined to the nearest wall by
              the shortest straight wall (up to `extend` px); if it cannot be joined, it is removed
  inside out  "outside" (outdoor / background) pixels the fill left in the hole that do not connect to the outside
              of the building -> given to the room around them
  seal        where a room still touches the outside inside the hole, the outside pixels along the contact become a
              wall (the plan's thickness)
  openings    a door / window whose end meets no wall: the wall is drawn on to the nearest wall in line (up to
              `reach` px); if there is none, the opening is removed
  wall ends   a wall that ends in the middle of a room is drawn on straight to the next wall (up to `extend` px);
              if no wall lies ahead, the dangling piece is removed back to the wall it starts from
  room types  in an open area (no wall between its parts) the hole takes the room types the area already has
              outside the hole (nearest one; straight borders): no patches of another type without a wall. A room
              the fill closed with walls keeps its one type (round-3 clean-up)
  doors       a room closed by walls that no door reaches gets one door, of the plan's usual door length, in the
              longest straight stretch of wall it shares with a neighbouring room (preferring a room that is already
              reachable, and new walls over existing ones)

Only pixels inside the hole change (a door may cut an existing wall at the hole's edge). No ground truth, no model.
How it fits in: called at the end of src/novelty/pipeline.fill_region after the clean-up when the config has
`repair.use: true` (configs/novelty/pipeline_sound.yaml, "v4"). The checks it uses are the ones reported in the
evaluation, so its effect is ALSO measured with the project's own rule check (C1-C18) and pictures.
"""

import numpy as np
from scipy import ndimage

from src.novelty import soundness as snd
from src.stage2.data.classes16 import C16

WALL, DOOR, WINDOW = C16.index["wall"], C16.index["door"], C16.index["window"]
BOUNDARY = C16.group("BOUNDARY")
ROOMS = C16.group("S")
OUT = C16.group("OUT")
SOLID = [WALL, DOOR, WINDOW]
K8 = np.ones((3, 3), bool)
K4 = ndimage.generate_binary_structure(2, 1)
DEFAULT = {"reach": 12, "extend": 80, "window_pad": 48, "rounds": 3, "remove": True,
           "remove_max": 24, "skip": []}   # remove_max: longer dangling pieces (px) are kept   # remove: dangling / floating walls may be removed; skip: repairs switched off
PROBLEMS = ("dead_ends", "floating", "bad_openings", "leaks", "doorless")   # what "sound" needs
ALSO = ("mixed",)                                 # also repaired, but never at the cost of the above


def problems(plan, region):
    r = snd.soundness(plan, region)
    return {k: r[k] for k in PROBLEMS + ALSO}


def total(p):
    return sum(p[k] for k in PROBLEMS)


def nearest_space(out, todo, keep=None):
    """Give every `todo` pixel the class of the nearest non-todo space pixel (rooms / outside)."""
    src = np.isin(out, ROOMS + OUT) & ~todo if keep is None else keep
    if not todo.any() or not src.any():
        return out
    _, (iy, ix) = ndimage.distance_transform_edt(~src, return_indices=True)
    out = out.copy()
    out[todo] = out[iy[todo], ix[todo]]
    return out


def nearest_room(out, todo):
    src = np.isin(out, ROOMS) & ~todo
    return nearest_space(out, todo, src) if src.any() else nearest_space(out, todo)


def door_length(plan):
    """The plan's usual door length (px): median long side of its doors, clipped to 8..16; 10 if it has none."""
    lab, n = ndimage.label(plan == DOOR, K8)
    sides = [max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) for sl in ndimage.find_objects(lab)]
    sides = [s for s in sides if s >= 5]
    return int(np.clip(np.median(sides), 8, 16)) if sides else 10


def wall_thickness(plan):
    from src.novelty.regularise import plan_thickness
    return plan_thickness(plan)


# ---------------------------------------------------------------- the single repairs (each returns a new map)

def fix_floating(out, hole, region, extend, T, remove=True):
    """Each wall group touching nothing outside the hole is joined to the nearest wall by the shortest straight
    wall (horizontal or vertical, the plan's thickness, through open space inside the hole, at most `extend` px);
    a group that cannot be joined is removed."""
    res = out.copy()
    solid = np.isin(out, SOLID)
    lab, _ = ndimage.label(solid, K8)
    bnd = np.isin(out, BOUNDARY)
    H, W = out.shape
    for k in snd.floating(out, region):
        G = lab == k
        edge = G & ndimage.binary_dilation(~G, K4)
        best = None
        for y, x in np.argwhere(edge & (out == WALL)):
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if 0 <= y + dy < H and 0 <= x + dx < W and G[y + dy, x + dx]:
                    continue                              # inward
                across = _cross(G, y, x, dx != 0, T)      # the piece's own cross-section at this edge
                if not across:
                    continue
                band = [(c, x) if dx != 0 else (y, c) for c in across]
                steps = []
                for a, b in band:
                    s = None
                    for n in range(1, extend + 1):
                        ty, tx = a + n * dy, b + n * dx
                        if not (0 <= ty < H and 0 <= tx < W) or not hole[ty, tx] and not bnd[ty, tx]:
                            break
                        if bnd[ty, tx] and not G[ty, tx]:
                            s = n
                            break
                        if bnd[ty, tx]:
                            break
                    steps.append(s)
                if any(s is None for s in steps):
                    continue
                n = max(steps)
                if best is None or n < best[0]:
                    best = (n, band, dy, dx)
        if best is not None:
            n, band, dy, dx = best
            for a, b in band:
                for m in range(1, n):
                    ty, tx = a + m * dy, b + m * dx
                    if not bnd[ty, tx]:
                        res[ty, tx] = WALL
        elif remove:
            res = nearest_space(res, G & hole)
    return res


def fix_inside_out(out, hole):
    outs = np.isin(out, OUT)
    lab, n = ndimage.label(outs, K4)
    todo = np.zeros(out.shape, bool)
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        comp = lab[sl] == k
        if not (comp & ~hole[sl]).any():                  # wholly inside the hole: enclosed by the building
            todo[sl] |= comp
    return nearest_room(out, todo) if todo.any() else out


def _march(free, y, x, dy, dx, n):
    """Steps from (y, x) along (dy, dx) while `free`; -> number of free steps before the first non-free pixel
    (or None if more than n / the map's edge)."""
    H, W = free.shape
    for k in range(1, n + 1):
        ty, tx = y + k * dy, x + k * dx
        if not (0 <= ty < H and 0 <= tx < W):
            return None
        if not free[ty, tx]:
            return k - 1
    return None


def fix_openings(out, hole, region, reach):
    """Each bad opening in the hole: draw its wall on to the next wall in line, else remove it."""
    res = out.copy()
    bnd = np.isin(out, BOUNDARY)
    for sl, comp, ok in snd.openings_status(out, region):
        if ok:
            continue
        ys, xs = np.nonzero(comp)
        ys, xs = ys + sl[0].start, xs + sl[1].start
        if not hole[ys, xs].any():
            continue
        whole = hole[ys, xs].all()                        # an opening reaching outside the hole is never removed
        horiz = (xs.max() - xs.min()) >= (ys.max() - ys.min())
        band = range(ys.min(), ys.max() + 1) if horiz else range(xs.min(), xs.max() + 1)
        done = True
        for end, step in ((xs.min() if horiz else ys.min(), -1), (xs.max() if horiz else ys.max(), 1)):
            beyond = [(c, end + step) if horiz else (end + step, c) for c in band]
            if any(0 <= y < out.shape[0] and 0 <= x < out.shape[1] and bnd[y, x] for y, x in beyond):
                continue                                  # this end already meets a wall
            gaps = []
            for c in band:
                y, x = (c, end) if horiz else (end, c)
                g = _march(~bnd, y, x, 0 if horiz else step, step if horiz else 0, reach)
                gaps.append(g)
            if any(g is None for g in gaps):
                done = False
                break
            for c, g in zip(band, gaps):
                for k in range(1, g + 1):
                    y, x = (c, end + k * step) if horiz else (end + k * step, c)
                    if hole[y, x]:
                        res[y, x] = WALL
        if not done and whole:
            m = np.zeros(out.shape, bool)
            m[ys, xs] = True
            res = nearest_space(res, m)
    return res


def fix_dead_ends(out, hole, region, extend, T, remove=True, remove_max=999):
    """Each dead end in the hole: extend straight to the next wall, else remove the dangling piece."""
    res = out.copy()
    ends, _ = snd.wall_ends(out, region)
    wall = out == WALL
    bnd = np.isin(out, BOUNDARY)
    for y, x in ends:
        if not hole[y, x]:
            continue
        # direction: away from the wall pixels behind the end
        win = wall[max(0, y - 4):y + 5, max(0, x - 4):x + 5]
        wy, wx = np.nonzero(win)
        dy, dx = y - (wy.mean() + max(0, y - 4)), x - (wx.mean() + max(0, x - 4))
        sy, sx = (0, int(np.sign(dx))) if abs(dx) >= abs(dy) else (int(np.sign(dy)), 0)
        if sy == 0 and sx == 0:
            continue
        # the wall's cross-section at the end
        cy, cx = y, x
        while 0 <= cy + sy < wall.shape[0] and 0 <= cx + sx < wall.shape[1] and wall[cy + sy, cx + sx]:
            cy, cx = cy + sy, cx + sx
        across = _cross(wall, cy, cx, sy == 0, T)
        if not across:
            continue
        gaps = []
        for a in across:
            py, px = (a, cx) if sy == 0 else (cy, a)
            gaps.append(_march(~bnd, py, px, sy, sx, extend))
        if all(g is not None for g in gaps) and max(gaps) - min(gaps) <= T + 2:
            g = max(gaps)
            for a in across:
                for k in range(1, g + 1):
                    py, px = ((a, cx + k * sx) if sy == 0 else (cy + k * sy, a))
                    if 0 <= py < wall.shape[0] and 0 <= px < wall.shape[1] and hole[py, px] and not bnd[py, px]:
                        res[py, px] = WALL
        elif remove:                                      # nothing ahead: remove the piece back to its junction
            piece = _dangling(wall & hole, y, x, -sy, -sx, T)
            if piece is not None and piece.sum() <= remove_max * T:
                res = nearest_space(res, piece)
    return res


def _cross(wall, cy, cx, vertical, T):
    """Coordinates across the wall at (cy, cx): rows (vertical=True, for a horizontal wall) or columns of the
    contiguous wall run through it, at most T + 2 long; [] if thicker (a junction)."""
    line = wall[:, cx] if vertical else wall[cy, :]
    c = cy if vertical else cx
    a0, a1 = c, c
    while a0 - 1 >= 0 and line[a0 - 1]:
        a0 -= 1
    while a1 + 1 < len(line) and line[a1 + 1]:
        a1 += 1
    return list(range(a0, a1 + 1)) if a1 - a0 + 1 <= T + 2 else []


def _dangling(wall, y, x, dy, dx, T):
    """Wall pixels of the straight piece ending at (y, x), walking back along (dy, dx) until a junction (a wall
    beside the piece) or 60 px; None if none found."""
    H, W = wall.shape
    m = np.zeros(wall.shape, bool)
    cy, cx = y, x
    for k in range(60):
        if not (0 <= cy < H and 0 <= cx < W) or not wall[cy, cx]:
            return m if m.any() else None
        # cross-section
        if dy == 0:
            a0, a1 = cy, cy
            while a0 - 1 >= 0 and wall[a0 - 1, cx]:
                a0 -= 1
            while a1 + 1 < H and wall[a1 + 1, cx]:
                a1 += 1
            if a1 - a0 + 1 > T + 2:
                return m if m.any() else None             # reached a crossing wall
            m[a0:a1 + 1, cx] = True
        else:
            a0, a1 = cx, cx
            while a0 - 1 >= 0 and wall[cy, a0 - 1]:
                a0 -= 1
            while a1 + 1 < W and wall[cy, a1 + 1]:
                a1 += 1
            if a1 - a0 + 1 > T + 2:
                return m if m.any() else None
            m[cy, a0:a1 + 1] = True
        cy, cx = cy + dy, cx + dx
    return None


def fix_doors(out, hole, region, Ld, T):
    """Give each doorless room touching the hole one door in a wall it shares with a neighbouring space."""
    res = out.copy()
    bnd = np.isin(out, BOUNDARY)
    lab, n = ndimage.label(~bnd, K4)
    H, W = out.shape
    doorish = ndimage.binary_dilation(out == DOOR, K8, iterations=2)
    reach = {k: bool(doorish[lab == k].any()) for k in range(1, n + 1)}
    edge_ids = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        comp = lab[sl] == k
        if comp.sum() < snd.MIN_ROOM or not region[sl][comp].any() or k in edge_ids or reach[k]:
            continue
        vals = out[sl][comp]
        if np.isin(vals, OUT).mean() > 0.2 or np.isin(vals, ROOMS).mean() < 0.5:
            continue
        best = None
        R = lab == k
        for sy, sx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            # wall pixels next to the room on that side, and what lies across the wall
            ys, xs = np.nonzero(R & np.roll(np.roll(out == WALL, -sy, 0), -sx, 1))
            cand = {}
            for y, x in zip(ys, xs):
                wy, wx = y + sy, x + sx
                t = _march(out == WALL, wy - sy, wx - sx, sy, sx, T + 3)
                if t is None or t == 0:
                    continue
                qy, qx = wy + (t - 1) * sy + sy, wx + (t - 1) * sx + sx
                if not (0 <= qy < H and 0 <= qx < W) or bnd[qy, qx] or lab[qy, qx] in (0, k):
                    continue
                N = lab[qy, qx]
                pos = x if sy else y                      # position along the wall
                cand.setdefault(N, []).append((pos, wy, wx, t))
            for N, pts in cand.items():
                pts.sort()
                runs, cur = [], [pts[0]]
                for p in pts[1:]:
                    if p[0] == cur[-1][0] + 1 and p[3] == cur[-1][3]:
                        cur.append(p)
                    else:
                        runs.append(cur)
                        cur = [p]
                runs.append(cur)
                for r in runs:
                    if len(r) < Ld + 2:
                        continue
                    mid = len(r) // 2
                    span = r[mid - Ld // 2: mid - Ld // 2 + Ld]
                    in_hole = np.mean([hole[p[1], p[2]] for p in span])
                    nvals = out[lab == N]
                    room_like = np.isin(nvals, ROOMS).mean() > 0.5
                    score = (2 if reach.get(N) else 0) + (1 if room_like else 0) + in_hole + min(len(r), 60) / 100
                    if best is None or score > best[0]:
                        best = (score, span, (sy, sx))
        if best is None:
            continue
        _, span, (sy, sx) = best
        for pos, wy, wx, t in span:
            for j in range(t):
                py, px = wy + j * sy, wx + j * sx
                if res[py, px] == WALL:
                    res[py, px] = DOOR
    return res


def fix_seal(out, hole, T):
    """Close the places in the hole where a room touches the outside: the outside pixels along the contact become
    wall, T px deep (only inside the hole)."""
    rooms, outs = np.isin(out, ROOMS), np.isin(out, OUT)
    contact = outs & ndimage.binary_dilation(rooms, K4) & hole
    if not contact.any():
        return out
    band = ndimage.binary_dilation(contact, K8, iterations=max(0, T - 1)) & outs & hole
    res = out.copy()
    res[band] = WALL
    return res


def fix_mixed(out, hole, enclosed=0.5, min_area=200):
    """One open area, its own room types: in a space with no wall between its parts, a patch of a room type that the
    fill drew in the hole takes the type of the nearest room-type pixel of that space OUTSIDE the hole (straight
    borders, no loose patches) - UNLESS walls enclose at least `enclosed` of its border and it has `min_area` px:
    then it is a room open on one side (open-plan kitchen, hallway) and stays. A space lying wholly in the hole (a
    room the fill closed) keeps its one type (round-3 clean-up)."""
    barrier = np.isin(out, BOUNDARY)
    lab, n = ndimage.label(~barrier, K4)
    res = out.copy()
    for k in np.unique(lab[hole & (lab > 0)]):
        comp = lab == k
        known = comp & ~hole & np.isin(out, ROOMS)
        if not known.any():
            continue
        todo = np.zeros(out.shape, bool)
        for cls in np.unique(out[comp & hole & np.isin(out, ROOMS)]):
            pl, pn = ndimage.label(comp & (out == cls), K4)
            for q in np.unique(pl[hole & (pl > 0)]):
                P = pl == q
                if (P & ~hole).any() and (P & ~hole).sum() >= 0.5 * P.sum():
                    continue                              # mostly an existing room: stays
                ring = ndimage.binary_dilation(P, K4) & ~P
                if P.sum() >= min_area and barrier[ring].mean() >= enclosed:
                    continue                              # a room open on one side: stays
                todo |= P & hole
        if todo.any():
            res = nearest_space(res, todo, known & ~todo)
    return res


# ---------------------------------------------------------------- driver

def repair_window(b, a, h, rc, T, Ld):
    """Repair one window (before / after / hole) -> (map, info)."""
    region = h
    out = a.copy()
    p0 = problems(out, region)
    log = []
    steps = (("floating", lambda o: fix_floating(o, h, region, rc["extend"], T, rc["remove"])),
             ("leaks", lambda o: fix_inside_out(o, h)),
             ("seal", lambda o: fix_seal(o, h, T)),
             ("bad_openings", lambda o: fix_openings(o, h, region, rc["reach"])),
             ("dead_ends", lambda o: fix_dead_ends(o, h, region, rc["extend"], T, rc["remove"], rc["remove_max"])),
             ("doorless", lambda o: fix_doors(o, h, region, Ld, T)),
             ("mixed", lambda o: fix_mixed(o, h)))
    cur = p0
    for _ in range(rc["rounds"]):
        changed = False
        for name, fn in steps:
            if name in rc["skip"]:
                continue
            if cur["leaks" if name == "seal" else name] == 0 and name != "leaks":
                continue
            new = fn(out)
            outside = ~h & (new != out)                   # only the hole changes; a door may cut the wall at its edge
            if name == "doorless":
                outside &= ~(ndimage.binary_dilation(h, K8, iterations=T + 2) & (new == DOOR))
            new[outside] = out[outside]
            if (new == out).all():
                continue
            p = problems(new, region)
            key = "leaks" if name == "seal" else name
            if total(p) <= total(cur) and (p[key] < cur[key] or (name == "leaks" and total(p) < total(cur))):
                out, cur, changed = new, p, True
                log.append(name)
            elif name == "leaks" and total(p) <= total(cur):
                out, cur = new, p                         # enclosed "outside" bits: always given to the room
        if not changed:
            break
    info = {"before": p0, "after": cur, "fixes": log}
    return out, info


def repair_fill(before, after, hole, rc=None):
    """Repair one region fill (see the module text) on a window around its hole -> (new map, info)."""
    rc = {**DEFAULT, **(rc or {})}
    H, W = before.shape
    ys, xs = np.nonzero(hole)
    p = rc["window_pad"]
    sl = (slice(max(0, ys.min() - p), min(H, ys.max() + p + 1)), slice(max(0, xs.min() - p), min(W, xs.max() + p + 1)))
    win, info = repair_window(before[sl], after[sl], hole[sl], rc, wall_thickness(before), door_length(before))
    out = after.copy()
    out[sl] = win
    return out, info
