"""Novelty round 3 - VALIDITY: clean-up of a region fill, so that what the system adds obeys basic drawing rules.

    new_map, info = regularise(before, after, hole)        # before / after the large-hole fill, hole = its box

Why: the round-2 fill was tuned for "are the missing walls back" and its raw output breaks rules any drawing obeys:
walls of changing thickness, ragged or curved wall pieces, loose blobs, small patches of another room type. This
step keeps the fill's IDEAS (where a wall goes, which room it is) and redraws them by rule:

  walls     the walls after the fill (existing + added) are read as straight pieces (horizontal, vertical, or a
            straight diagonal). Every piece the fill adds to is redrawn with ONE thickness: that of the existing wall
            it is part of or continues, else the plan's usual wall thickness. Ends are extended up to `snap` px to
            meet a wall, pieces in line are joined, corners are closed. Dropped: pieces shorter than `min_len`,
            curved pieces, blobs, pieces that only thicken a full existing wall, the shorter of two pieces that
            run side by side closer than `min_gap`, and pieces not connected to the existing walls.
  wall ends a wall must end at a wall: a free end runs straight on to the next wall; if the gap to it is door-sized
            and has a room on both sides it is a doorway and gets a door; if no wall lies ahead, the piece is cut
            back to the last wall it touches.
  openings  an added door / window is kept only inside a wall line (wall on both ends) and gets that wall's
            thickness; otherwise it is removed.
  fixtures  an added fixture is kept only if it is not a speck and stands at a wall.
  rooms     a space closed by walls, doors, windows or an open doorway (a wall end facing a wall at most `doorway` px
            away) that lies mostly inside the filled area is a new room and gets ONE type (majority vote of the
            fill). In open spaces, patches of a room type smaller than `min_room` px are absorbed by their
            neighbours (real plans do have large open-plan zones without a wall, so large zones stay).
  existing  what the fill keeps of the map before stays as it is; what the fill removes stays removed (the parser
            draws stray bits of wall and door on erased paper, and the fill is trained to take them out).

How it fits in: called at the end of src/novelty/pipeline.fill_region when the config has `regularise.use: true`
(configs/novelty/pipeline_valid.yaml, "v3d"). Pure geometry on label maps: no model, no training, nothing learned.
`checks()` are the simple validity counts reported next to the accuracy numbers.
"""

import cv2
import numpy as np
from scipy import ndimage

from src.stage2.data.classes16 import C16

WALL, DOOR, WINDOW = C16.index["wall"], C16.index["door"], C16.index["window"]
BOUNDARY = C16.group("BOUNDARY")
ROOMS = C16.group("S")
SPACE = ROOMS + C16.group("OUT")
FIX = C16.group("F")
DEFAULT = {"min_len": 8, "snap": 6, "t_max": 8, "min_new": 6, "doorway": 20, "min_fixture": 6, "min_opening": 4,
           "min_room": 60, "min_gap": 8, "door_min": 6, "door_max": 27, "window_pad": 48}


def grown(m, r):
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1,) * 2, np.uint8)) > 0


def plan_thickness(label):
    """The plan's usual wall thickness in px (3 to 5): the median over its wall pixels."""
    wall = np.pad(np.ascontiguousarray(label == WALL, np.uint8), 10)   # (padded: the map's edge is not wall)
    if wall.sum() < 20:
        return 3
    across, along = np.zeros(wall.shape, np.uint8), np.zeros(wall.shape, np.uint8)
    for k in range(1, 10):                                # run length through each wall pixel, both ways (up to 9)
        across += axis_runs(wall.T, k).T
        along += axis_runs(wall, k)
    return int(np.clip(np.median(np.minimum(across, along)[wall > 0]), 3, 5))   # thickness = the shorter of the two


# ---------------------------------------------------------------- walls -> straight pieces

def axis_runs(m, L):
    """Pixels of `m` that lie on a horizontal run of at least L px."""
    k = np.ones((1, L), np.uint8)                         # (an exact opening: OpenCV's own shifts by 1 px for even L)
    e = cv2.erode(np.ascontiguousarray(m, np.uint8), k, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.dilate(e, k, anchor=(L - 1, 0), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0


def axis_segments(runs, W0, L, o, P=None):
    """Horizontal straight pieces of a mask of NEW wall pixels that lie on straight runs
    -> [{o, c (centre row), a0, a1 (columns), t (thickness), new (pixels), old_share, mask}].
    old_share = share of the piece's columns that have an existing wall (W0) within 2 px above or below it (the
    piece only thickens that wall); old_t = that wall's thickness; c_both = centre row of the two together. A run whose centre row drifts by more than 2 px is cut there.
    P: the structure net's walls; a piece it drew is centred on its line."""
    n, comp = cv2.connectedComponents(np.ascontiguousarray(runs, np.uint8), connectivity=8)
    rows = np.arange(runs.shape[0])[:, None]
    beside = cv2.dilate(np.ascontiguousarray(W0, np.uint8), np.ones((5, 1), np.uint8)) > 0
    out = []
    for i in range(1, n):
        m = comp == i
        cols = np.nonzero(m.any(0))[0]
        x0, x1 = cols.min(), cols.max() + 1
        m = m[:, x0:x1]
        th = m.sum(0)
        centre = (m * rows).sum(0) / np.maximum(th, 1)
        j0 = 0
        for j in range(1, len(th) + 1):
            if j < len(th) and abs(centre[j] - np.mean(centre[j0:j])) <= 2:
                continue
            if j - j0 >= L:
                piece = m[:, j0:j]
                s = {"o": o, "c": float(np.median(centre[j0:j])), "a0": int(x0 + j0), "a1": int(x0 + j - 1),
                     "t": int(round(float(np.median(th[j0:j])))), "new": int(piece.sum()), "host": None,
                     "old_share": float((piece & beside[:, x0 + j0:x0 + j]).any(0).mean())}
                near = W0[:, x0 + j0:x0 + j] & beside_new(piece)     # the existing wall right beside the piece
                nc = near.any(0)
                s["old_t"] = float(np.median(near.sum(0)[nc])) if nc.any() else 0.0
                both = piece | near
                s["c_both"] = float(np.median((both * rows).sum(0) / both.sum(0)))
                if P is not None:                                    # the structure net's own line, if it drew this wall
                    pr = piece & P[:, x0 + j0:x0 + j]
                    pc = pr.any(0)
                    if pc.sum() >= max(L, 0.5 * (j - j0)):
                        s["c"] = float(np.median((pr * rows).sum(0)[pc] / pr.sum(0)[pc]))
                s["mask"] = np.zeros(runs.shape, bool)               # the pixels this piece was read from
                s["mask"][:, x0 + j0:x0 + j] = piece
                out.append(s)
            j0 = j
    return out


def beside_new(piece):
    """Pixels within 2 px above / below a piece (same columns)."""
    return cv2.dilate(np.ascontiguousarray(piece, np.uint8), np.ones((5, 1), np.uint8)) > 0


def straight_leftovers(left, L, t_max):
    """Added wall pixels on no horizontal / vertical run: a thin straight strip becomes a piece (axis-aligned if it
    is within 15 degrees of an axis, else a diagonal); curved pieces and blobs are dropped."""
    n, comp = cv2.connectedComponents(np.ascontiguousarray(left, np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        ys, xs = np.nonzero(comp == i)
        if len(ys) < L:
            continue
        pts = np.stack([ys, xs], 1).astype(np.float64)
        mean = pts.mean(0)
        val, vec = np.linalg.eigh(np.cov((pts - mean).T))
        long, wide = np.sqrt(12 * max(val[1], 0)), np.sqrt(12 * max(val[0], 0))
        if long < 1.5 * L or wide > min(t_max, 0.35 * long):
            continue
        d = vec[:, 1]                                        # direction (dy, dx) of the strip
        extra = {"t": 0, "new": len(ys), "host": None, "old_share": 0.0, "old_t": 0.0, "mask": comp == i}
        if abs(d[1]) >= np.cos(np.radians(15)):
            out.append({"o": "h", "c": float(mean[0]), "a0": int(xs.min()), "a1": int(xs.max()), **extra})
        elif abs(d[0]) >= np.cos(np.radians(15)):
            out.append({"o": "v", "c": float(mean[1]), "a0": int(ys.min()), "a1": int(ys.max()), **extra})
        else:
            p0, p1 = mean - d * long / 2, mean + d * long / 2
            out.append({"o": "d", "p": ((int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1])))),
                        **extra})
    return out


def rows_of(s):
    """First row (or column) and one past the last of an axis piece's band."""
    r0 = int(round(s["c"] - (s["t"] - 1) / 2))
    return r0, r0 + s["t"]


def raster(s, shape):
    m = np.zeros(shape, np.uint8)
    if s["o"] == "d":
        (y0, x0), (y1, x1) = s["p"]
        cv2.line(m, (x0, y0), (x1, y1), 1, thickness=max(1, s["t"]))
        return m > 0
    r0, r1 = rows_of(s)
    r0, a0 = max(0, r0), max(0, s["a0"])
    if s["o"] == "h":
        m[r0:r1, a0:s["a1"] + 1] = 1
    else:
        m[a0:s["a1"] + 1, r0:r1] = 1
    return m > 0


def length(s):
    if s["o"] == "d":
        (y0, x0), (y1, x1) = s["p"]
        return float(np.hypot(y1 - y0, x1 - x0))
    return s["a1"] - s["a0"] + 1


def merge_in_line(segs, snap):
    """Axis pieces on the same line (centres within 3 px: their bands touch) with a gap of at most `snap` px become one piece."""
    segs = [dict(s) for s in segs]
    changed = True
    while changed:
        changed = False
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                a, b = segs[i], segs[j]
                if a["o"] != b["o"] or a["o"] == "d" or abs(a["c"] - b["c"]) > 3:
                    continue
                if max(a["a0"], b["a0"]) - min(a["a1"], b["a1"]) - 1 > snap:
                    continue
                la, lb = length(a), length(b)
                segs[i] = {"o": a["o"], "c": (a["c"] * la + b["c"] * lb) / (la + lb), "a0": min(a["a0"], b["a0"]),
                           "a1": max(a["a1"], b["a1"]), "t": a["t"] if la >= lb else b["t"], "new": a["new"] + b["new"],
                           "host": a["host"] or b["host"], "old_share": (a["old_share"] * la + b["old_share"] * lb) / (la + lb), "old_t": max(a["old_t"], b["old_t"]),
                           "mask": a["mask"] | b["mask"]}
                del segs[j]
                changed = True
                break
            if changed:
                break
    return segs


def drop_doubles(segs, gap):
    """Two added pieces that run side by side closer than `gap` px (centre to centre) over at least half of the
    shorter one: the space between them is no room, the fill drew one wall twice. The shorter piece is dropped."""
    segs = sorted(segs, key=length, reverse=True)
    kept = []
    for s in segs:
        twin = any(k["o"] == s["o"] != "d" and abs(k["c"] - s["c"]) < gap and
                   min(k["a1"], s["a1"]) - max(k["a0"], s["a0"]) + 1 >= 0.5 * length(s) for k in kept)
        if not twin:
            kept.append(s)
    return kept


def join_corners(segs, snap):
    """A horizontal and a vertical piece whose ends come within `snap` px of each other meet in a clean corner:
    the end is moved onto the other piece (extended, or an overshoot cut)."""
    for h in [s for s in segs if s["o"] == "h"]:
        for v in [s for s in segs if s["o"] == "v"]:
            th, tv = max(h["t"], 1), max(v["t"], 1)
            if not (h["a0"] - snap - tv <= v["c"] <= h["a1"] + snap + tv and
                    v["a0"] - snap - th <= h["c"] <= v["a1"] + snap + th):
                continue
            if abs(h["a1"] - v["c"]) <= snap + tv:
                h["a1"] = int(round(v["c"] + tv / 2))
            elif abs(h["a0"] - v["c"]) <= snap + tv:
                h["a0"] = int(round(v["c"] - tv / 2))
            if abs(v["a1"] - h["c"]) <= snap + th:
                v["a1"] = int(round(h["c"] + th / 2))
            elif abs(v["a0"] - h["c"]) <= snap + th:
                v["a0"] = int(round(h["c"] - th / 2))
    return segs


def meet_existing(s, W, snap, t_max, L):
    """One horizontal piece against the existing walls W (for a vertical piece: pass W.T). Each end is extended up
    to `snap` px until it touches a wall. If that wall runs on in the same direction, its (centre, thickness) is
    returned (the piece continues that wall); else None."""
    H, Wd = W.shape
    found = None
    for end, step in (("a1", 1), ("a0", -1)):
        for k in range(1, snap + 1):
            x = s[end] + step * k
            if not 0 <= x < Wd:
                break
            r0, r1 = max(0, int(round(s["c"])) - 2), min(H, int(round(s["c"])) + 3)
            rows = np.nonzero(W[r0:r1, x])[0]
            if not len(rows):
                continue
            y = r0 + int(rows[np.argmin(np.abs(r0 + rows - s["c"]))])
            a = b = y
            while a > 0 and W[a - 1, x]:
                a -= 1
            while b < H - 1 and W[b + 1, x]:
                b += 1
            s[end] = x - step
            ahead = W[y, x:x + L] if step > 0 else W[y, max(0, x - L + 1):x + 1]
            if b - a + 1 <= t_max and ahead.sum() >= L - 1:           # a wall in line with the piece
                found = ((a + b) / 2, b - a + 1)
            break
    return found


def wall_pieces(U, W0, T, rc, prior=None):
    """All wall pixels after the fill `U` (existing W0 + added) -> the straight pieces of ADDED wall, redrawn by the
    wall rules (see the module text)."""
    L, snap, t_max = rc["min_len"], rc["snap"], rc["t_max"]
    U = (cv2.morphologyEx(U.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0) | W0
    hr, vr = axis_runs(U, L), axis_runs(U.T, L).T                    # straight runs of existing + added wall together
    vs = axis_segments((vr & ~W0).T, W0.T, L, "v", None if prior is None else prior.T)
    for s in vs:
        s["mask"] = s["mask"].T
    segs = axis_segments(hr & ~W0, W0, L, "h", prior) + vs
    segs = [s for s in segs if s["new"] >= rc["min_new"] and s["t"] < rc.get("blob", L)   # thick: a blob
            and (s["old_share"] < 0.5 or s["old_t"] < T)]            # it only thickens a full existing wall: not drawn
    for s in segs:                                                   # it completes a wall the parser drew too thin:
        if s["old_share"] >= 0.5:                                    # one wall of the usual thickness on their common
            s["c"] = s["c_both"]                                     # centre line

    lab, n = ndimage.label(U & ~W0, structure=np.ones((3, 3)))        # an added shape that is mostly NOT on straight
    total = np.bincount(lab.ravel(), minlength=n + 1)                # runs (an arc, a ring) is no wall: the short
    on = np.bincount(lab[(hr | vr) & (lab > 0)], minlength=n + 1)    # straight bits of it are not kept either
    curved = np.isin(lab, [k for k in range(1, n + 1) if on[k] < 0.6 * total[k]])
    segs = [s for s in segs if (s["mask"] & curved).sum() <= 0.5 * s["new"] or not rc.get("curved", True)]
    segs += straight_leftovers(U & ~hr & ~vr & ~W0, L, t_max)
    segs = join_corners(drop_doubles(merge_in_line(segs, snap), rc["min_gap"]), snap)
    kept = []
    for s in segs:
        if s["o"] != "d":
            hit = meet_existing(s, W0 if s["o"] == "h" else W0.T, snap, t_max, L)   # the wall this piece continues
            s["c"], s["t"] = (hit[0], int(np.clip(hit[1], T, T + 2))) if hit else (s["c"], T)
        else:
            s["t"] = T
        if length(s) >= L:
            kept.append(s)
    while kept and rc.get("anchor", True):                           # only pieces connected to the existing walls
        rs = [raster(s, W0.shape) for s in kept]
        lab, _ = ndimage.label(grown(W0 | np.any(rs, 0), 1), structure=np.ones((3, 3)))
        anchored = set(np.unique(lab[W0]))
        ok = [bool(set(np.unique(lab[r])) & anchored) for r in rs]
        if all(ok):
            break
        kept = [s for s, k in zip(kept, ok) if k]
    return kept


def open_ends(segs, base):
    """Free ends of axis pieces: [(piece, end name, step)]. An end is held if another wall (`base` = the existing
    walls and doors, or another piece) touches the piece's band at its last pixel or right beyond it."""
    rs = [raster(s, base.shape) for s in segs]
    count = np.sum(rs, 0) if rs else np.zeros(base.shape, int)
    out = []
    for s, r in zip(segs, rs):
        if s["o"] == "d":
            continue
        others = base | (count - r > 0)
        W = others if s["o"] == "h" else others.T
        r0, r1 = rows_of(s)
        for end, step in (("a1", 1), ("a0", -1)):
            x0, x1 = sorted((s[end], s[end] + step))
            if not W[max(0, r0 - 1):r1 + 1, max(0, x0):x1 + 1].any():
                out.append((s, end, step))
    return out


def finish_ends(segs, W0, h, a, rc):
    """A wall must end at a wall. Every free end of an added piece looks straight on (inside the filled area and
    up to door_max px beyond it):
      * a wall is met and the gap is door-sized (door_min .. door_max px) with a room on both sides: a doorway,
        closed by a DOOR in the wall line;
      * a wall is met otherwise: the piece runs on to that wall;
      * no wall is met: the piece is cut back to the last wall it touches (a stub that leads nowhere goes).
    Returns (pieces, bool map of the new doors). Repeated, because cutting one piece can free another."""
    L = rc["min_len"]
    doors = np.zeros(W0.shape, bool)
    reach = grown(h, rc["door_max"])
    for _ in range(6):
        walls = W0.copy()
        for s in segs:
            walls |= raster(s, W0.shape)
        ends = open_ends(segs, W0 | doors)
        if not ends:
            break
        for s, end, step in ends:
            W, R, D, A = (walls, reach, doors, a) if s["o"] == "h" else (walls.T, reach.T, doors.T, a.T)
            r0, r1 = rows_of(s)
            r0, mid = max(0, r0), int(np.clip(round(s["c"]), 0, W.shape[0] - 1))
            x, hit = s[end] + step, None
            while 0 <= x < W.shape[1] and R[mid, x]:
                if W[r0:r1, x].any():
                    hit = x
                    break
                x += step
            if hit is not None:
                lo, hi = sorted((s[end] + step, hit - step))
                sides = np.isin(A[max(0, r0 - 3), lo:hi + 1], ROOMS).all() and \
                    np.isin(A[min(W.shape[0] - 1, r1 + 2), lo:hi + 1], ROOMS).all()
                if rc["door_min"] <= hi - lo + 1 <= rc["door_max"] and sides:
                    D[r0:r1, lo:hi + 1] = True
                else:
                    s[end] = hit - step
            else:                                                    # cut back to the last wall it touches
                O = walls & ~raster(s, W0.shape)
                O = O if s["o"] == "h" else O.T
                cols = np.nonzero(O[max(0, r0 - 1):r1 + 1].any(0))[0]
                cols = cols[(cols >= s["a0"]) & (cols <= s["a1"])]
                if step > 0:
                    s["a1"] = int(cols.max()) if len(cols) else s["a0"] - 1
                else:
                    s["a0"] = int(cols.min()) if len(cols) else s["a1"] + 1
        segs = [s for s in segs if s["o"] == "d" or length(s) >= L]
    return segs, doors


def doorway_barriers(segs, walls, reach, base):
    """Open doorways: from every free wall end, straight on; if a wall is met within `reach` px, the pixels in
    between close the room for the room rule (they are NOT drawn as wall)."""
    out = np.zeros(walls.shape, bool)
    for s, end, step in open_ends(segs, base):
        W, O = (walls, out) if s["o"] == "h" else (walls.T, out.T)
        r0, r1 = rows_of(s)
        r0 = max(0, r0)
        for k in range(1, reach + 1):
            x = s[end] + step * k
            if not 0 <= x < W.shape[1]:
                break
            if W[r0:r1, x].any():
                lo, hi = sorted((s[end] + step, x - step))
                O[r0:r1, lo:hi + 1] = True
                break
    return out


# ---------------------------------------------------------------- openings, fixtures, rooms

def opening_box(comp, walls, t_max, T, min_len):
    """An added door / window blob -> its box inside the wall line (y0, x0, y1, x1), or None if it is not in a wall
    (no wall within 3 px beyond both of its ends)."""
    ys, xs = np.nonzero(comp)
    flip = (ys.max() - ys.min()) > (xs.max() - xs.min())             # a vertical opening: work on the transpose
    W = walls.T if flip else walls
    if flip:
        ys, xs = xs, ys
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    if x1 - x0 + 1 < min_len:
        return None
    rows = slice(max(0, y0 - 1), y1 + 2)
    sides = []
    for cols in (range(x0 - 1, max(-1, x0 - 4), -1), range(x1 + 1, min(W.shape[1], x1 + 4))):
        hit = next((x for x in cols if W[rows, x].any()), None)
        if hit is None:
            return None
        ry = rows.start + np.nonzero(W[rows, hit])[0]
        a, b = ry.min(), ry.max()
        while a > 0 and W[a - 1, hit]:
            a -= 1
        while b < W.shape[0] - 1 and W[b + 1, hit]:
            b += 1
        sides.append((hit, a, b))
    (xl, al, bl), (xr, ar, br) = sides
    bands = [(a, b) for a, b in ((al, bl), (ar, br)) if b - a + 1 <= t_max]
    if bands:
        r0, r1 = bands[0]
    else:
        r0 = int(round((y0 + y1) / 2 - (T - 1) / 2))
        r1 = r0 + T - 1
    box = (int(r0), int(xl + 1), int(r1), int(xr - 1))
    if box[3] - box[1] + 1 < min_len:
        return None
    return (box[1], box[0], box[3], box[2]) if flip else box


def nearest_space(out, todo, src=None):
    """Pixels `todo` take the label of the nearest pixel of `src` (default: any pixel that holds a space class)."""
    src = (np.isin(out, SPACE) if src is None else src) & ~todo
    if not todo.any() or not src.any():
        return out
    idx = ndimage.distance_transform_edt(~src, return_distances=False, return_indices=True)
    out = out.copy()
    out[todo] = out[idx[0][todo], idx[1][todo]]
    return out


def clean_window(b, a, h, rc, T, prior=None):
    """before / after / hole on one window -> (cleaned map, info)."""
    snap, t_max = rc["snap"], rc["t_max"]
    W0 = (b == WALL) & (a == WALL)                                   # existing walls the fill kept
    keep_b = np.isin(b, BOUNDARY + FIX) & (a == b)                   # existing structure the fill kept: stays
    segs = wall_pieces(((a == WALL) & h) | W0, W0, T, rc, prior)
    door_new = np.zeros(b.shape, bool)
    if rc.get("finish_ends", True):
        segs, door_new = finish_ends(segs, W0, h, a, rc)
    new_wall = np.zeros(b.shape, bool)
    L = rc["min_len"]
    frag = {"h": W0 & ~axis_runs(W0, 3 * L) & ~axis_runs(W0.T, L).T,   # existing wall bits on no long run of their own
            "v": W0 & ~axis_runs(W0.T, 3 * L).T & ~axis_runs(W0, L)}
    local = grown((a == WALL) & h & ~W0, snap + 1) & grown(h, snap + 1)   # only near what the fill added
    for s in segs:
        r = raster(s, b.shape)
        new_wall |= r
        if rc.get("absorb", True) and s["o"] != "d":                 # short ragged bits of existing wall right
            keep_b &= ~(frag[s["o"]] & grown(r, 2) & ~r & local)     # beside a new wall are redrawn with it
    W0 = W0 & keep_b
    new_wall &= ~keep_b & ~grown(np.isin(b, [DOOR, WINDOW]) & (a == b), 1)
    lab, n = ndimage.label(new_wall, structure=np.ones((3, 3)))       # leftover specks of the masking above
    new_wall &= ~np.isin(lab, np.nonzero(np.bincount(lab.ravel()) < L)[0])
    out = a.copy()
    out[keep_b] = b[keep_b]
    todo = np.isin(out, BOUNDARY + FIX) & ~keep_b                    # everything the fill added is decided again
    out[new_wall] = WALL
    todo &= ~new_wall
    walls = W0 | new_wall
    door_new &= ~keep_b & ~new_wall
    out[door_new] = DOOR                                             # the doors of finish_ends
    todo &= ~door_new
    n_open, n_fix = ndimage.label(door_new)[1], 0
    for cls in (DOOR, WINDOW):                                       # openings: only inside a wall line
        lab, n = ndimage.label((a == cls) & h & (b != cls), structure=np.ones((3, 3)))
        for i in range(1, n + 1):
            box = opening_box(lab == i, walls, t_max, T, rc["min_opening"])
            if box is None:
                continue
            m = np.zeros(b.shape, bool)
            m[max(0, box[0]):box[2] + 1, max(0, box[1]):box[3] + 1] = True
            m &= ~(np.isin(b, [DOOR, WINDOW]) & (a == b)) & grown(h, snap + 1)
            out[m] = cls
            todo &= ~m
            n_open += 1
    fixed = np.isin(out, BOUNDARY) & ~todo                           # the structure that stays
    walls = (out == WALL) & fixed
    for cls in FIX:                                                  # fixtures: no specks, at a wall
        lab, n = ndimage.label((a == cls) & h & (b != cls) & ~fixed)
        for i in range(1, n + 1):
            m = lab == i
            if m.sum() >= rc["min_fixture"] and (grown(m, 2) & walls).any():
                out[m] = cls
                todo &= ~m
                n_fix += 1
    out = nearest_space(out, todo)
    # rooms: a closed new room gets one type; elsewhere room-type patches smaller than min_room are absorbed
    barrier = np.isin(out, BOUNDARY) | doorway_barriers(segs, walls, rc["doorway"], W0 | (out == DOOR))
    lab, n = ndimage.label(~barrier)
    edge = np.zeros(b.shape, bool)
    edge[0], edge[-1], edge[:, 0], edge[:, -1] = True, True, True, True
    n_rooms = 0
    for i in np.unique(lab[h & (lab > 0)]) if rc.get("rooms", True) else []:
        S = lab == i
        space = S & np.isin(out, SPACE)
        closed = not (S & edge).any() and (S & h).sum() >= 0.5 * S.sum()
        votes = out[space & np.isin(out, ROOMS)] if closed else np.empty(0, int)
        if len(votes):
            out[space] = np.bincount(votes).argmax()
            n_rooms += 1
            continue
        small = np.zeros(b.shape, bool)
        for cls in np.unique(out[space & h]):
            l2, n2 = ndimage.label(space & (out == cls))
            size = np.bincount(l2.ravel())
            ids = [k for k in np.unique(l2[h & (l2 > 0)]) if size[k] < rc["min_room"]]
            small |= np.isin(l2, ids) & h
        if small.any() and (space & ~small).any():
            out = nearest_space(out, small, space & ~small)
    info = {"walls": len(segs), "wall_px": int(new_wall.sum()), "openings": n_open, "fixtures": n_fix,
            "new_rooms": n_rooms, "open_ends": len(open_ends(segs, W0 | (out == DOOR)))}
    return out, info


def regularise(before, after, hole, rc=None, prior=None):
    """The clean-up of one region fill (see the module text) -> (new map, info). Works on a window around the hole.
    prior: bool map of the structure net's 3 px walls; a redrawn wall that it drew is put on its line (the fill's
    own wall there is a wide band, whose middle is a worse guess of the wall's place)."""
    rc = {**DEFAULT, **(rc or {})}
    H, W = before.shape
    ys, xs = np.nonzero(hole)
    p = rc["window_pad"]
    sl = (slice(max(0, ys.min() - p), min(H, ys.max() + p + 1)), slice(max(0, xs.min() - p), min(W, xs.max() + p + 1)))
    win, info = clean_window(before[sl], after[sl], hole[sl], rc, plan_thickness(before),
                             None if prior is None else prior[sl])
    out = after.copy()
    out[sl] = win
    return out, info


# ---------------------------------------------------------------- validity counts (for the reports)

def checks(before, plan, region, L=8):
    """Simple counts on what a plan ADDS inside `region`. The true layout does not score 0 on them (real drawings
    have walls of two thicknesses and open-plan zones), so read them next to the truth's own values.
      ragged     share of the added wall pixels that are not on a straight horizontal / vertical run of >= L px
      thick_sd   spread (standard deviation, px) of the thickness of the added walls
      loose      added wall groups not connected to any wall that was there before
      specks     added pieces (wall, door, window, fixture) smaller than 6 px
      patches    room-type patches smaller than 60 px that the plan creates in the region"""
    W0, W1 = before == WALL, plan == WALL
    added = W1 & region & ~W0
    out = {"added_wall_px": int(added.sum()), "ragged": 0.0, "thick_sd": 0.0, "loose": 0, "specks": 0, "patches": 0}
    if added.any():
        runs = axis_runs(W1, L) | axis_runs(W1.T, L).T
        out["ragged"] = round(float((added & ~runs).sum() / added.sum()), 3)
        dt = cv2.distanceTransform(W1.astype(np.uint8), cv2.DIST_L2, 3)
        ridge = (dt > 0) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-6) & added
        out["thick_sd"] = round(float(np.std(2 * dt[ridge])), 2) if ridge.any() else 0.0
        lab, n = ndimage.label(W1, structure=np.ones((3, 3)))
        out["loose"] = len(set(np.unique(lab[added])) - set(np.unique(lab[W0])))
    real = (plan != before) & region & np.isin(plan, BOUNDARY + FIX)
    lab, n = ndimage.label(real, structure=np.ones((3, 3)))
    out["specks"] = int((np.bincount(lab.ravel())[1:] < 6).sum()) if n else 0
    changed = (plan != before) & region & np.isin(plan, ROOMS)
    for cls in np.unique(plan[changed]):
        lab, n = ndimage.label(plan == cls)
        size = np.bincount(lab.ravel())
        out["patches"] += sum(1 for k in np.unique(lab[changed & (plan == cls)]) if size[k] < 60)
    return out
