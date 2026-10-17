"""Per-plan measurements on a 16-class label map (completion scale, walls = 3 px).

Each function measures one thing; build_cubicasa.py collects them over all
plans and report.py turns them into percentiles / the size constants.
Distances use the chessboard metric, because the constraint operators are
square max-pools: a square kernel of size 2r+1 reaches exactly r px.
"""

import cv2
import numpy as np

from src.stage2.data.classes16 import C16

SPACE_PAIR_CLASSES = C16.group("S") + C16.group("OD") + C16.group("BG")   # the 8 "spaces" of C6


def pixel_counts(label):
    return np.bincount(label.ravel(), minlength=C16.n)[: C16.n]


def components(mask):
    """Connected components (4-connectivity) -> (count, label image)."""
    n, comp = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    return n - 1, comp


def component_counts(label):
    return np.array([components(label == c)[0] for c in range(C16.n)])


def chess_dt(mask):
    """Chessboard distance of every pixel in `mask` to the nearest pixel outside it (border = outside)."""
    padded = np.pad(mask.astype(np.uint8), 1)
    return cv2.distanceTransform(padded, cv2.DIST_C, 3)[1:-1, 1:-1]


def square_widths(label, classes):
    """For each component of each class: the largest odd square that fits inside it (= 2*maxDT - 1).

    opn(X, k) with a k x k square removes a component exactly when this width is < k.
    Returns {class index: list of widths}.
    """
    out = {}
    for c in classes:
        n, comp = components(label == c)
        if n == 0:
            continue
        dt = chess_dt(label == c)
        best = np.zeros(n + 1)
        np.maximum.at(best, comp.ravel(), dt.ravel())
        out[c] = (2 * best[1:] - 1).astype(int).tolist()
    return out


def space_adjacency(label, max_gap):
    """C6 statistics: for pairs of different spaces (6 room types + outdoor + background),
    count horizontal/vertical pixel pairs that touch DIRECTLY and pairs separated only by
    boundary pixels (wall/door/window/railing) of thickness < max_gap.

    Returns (direct, total), each an (8, 8) symmetric count matrix in SPACE_PAIR_CLASSES order.
    """
    k = len(SPACE_PAIR_CLASSES)
    lut = np.full(256, -1, np.int64)
    lut[SPACE_PAIR_CLASSES] = np.arange(k)
    sp = lut[label]
    boundary = np.isin(label, C16.group("BOUNDARY"))
    direct = np.zeros((k, k), np.int64)
    total = np.zeros((k, k), np.int64)
    for a_map, b_bound in ((sp, boundary), (sp.T, boundary.T)):
        h, w = a_map.shape
        run = np.ones((h, w), bool)        # "all pixels strictly between are boundary"
        for d in range(1, max_gap + 1):
            if d >= w:
                break
            if d > 1:
                run = run[:, :-1] & b_bound[:, d - 1:]
            a, b = a_map[:, :w - d], a_map[:, d:]
            ok = (a >= 0) & (b >= 0) & (a != b) & run[:, : w - d]
            pairs = np.bincount(a[ok] * k + b[ok], minlength=k * k).reshape(k, k)
            total += pairs + pairs.T
            if d == 1:
                direct += pairs + pairs.T
    return direct, total


def fixture_rooms(before_fixtures, label):
    """(4 fixtures x 16) counts: which class lay under each fixture class's pixels."""
    fix = C16.group("F")
    out = np.zeros((len(fix), C16.n), np.int64)
    for i, f in enumerate(fix):
        out[i] = pixel_counts(before_fixtures[label == f])
    return out


def door_fixture_distances(label):
    """For each door component: chessboard distance from the door to the nearest fixture pixel."""
    fixtures = np.isin(label, C16.group("F"))
    if not fixtures.any():
        return []
    dist = chess_dt(~fixtures)          # distance of every non-fixture pixel to the nearest fixture
    n, comp = components(label == C16.index["door"])
    if n == 0:
        return []
    best = np.full(n + 1, np.inf)
    np.minimum.at(best, comp.ravel(), dist.ravel())
    return best[1:].tolist()


def fixture_room_distances(label):
    """For each fixture component: the largest distance from one of its pixels to the nearest room pixel."""
    rooms = np.isin(label, C16.group("S"))
    if not rooms.any():
        return []
    dist = chess_dt(~rooms)
    n, comp = components(np.isin(label, C16.group("F")))
    if n == 0:
        return []
    worst = np.zeros(n + 1)
    np.maximum.at(worst, comp.ravel(), dist.ravel())
    return worst[1:].tolist()


def pixel_open_hist(label, classes, kmax=64):
    """Per-pixel width statistics for C7: for every pixel of class c, the largest k such
    that it survives an opening with a k x k square (= it lies in a k x k square fully
    inside the class). Returns {c: histogram over k = 0..kmax} (kmax means ">= kmax").

    Like the torch erosion, the image border counts as inside (borderValue = max).
    """
    out = {}
    for c in classes:
        mask = (label == c).astype(np.uint8)
        if not mask.any():
            continue
        best = mask.astype(np.int32)                 # every pixel survives k = 1
        for k in range(2, kmax + 1):
            fits = cv2.erode(mask, np.ones((k, k), np.uint8), anchor=(0, 0))     # square with top-left here fits
            if not fits.any():
                break
            covered = cv2.dilate(fits, np.ones((k, k), np.uint8), anchor=(k - 1, k - 1)) > 0
            best[covered & (mask > 0)] = k
        out[c] = np.bincount(best[mask > 0], minlength=kmax + 1)[: kmax + 1]
    return out


def room_areas(label):
    """{room class: [area of each 4-connected room instance in px]}; fixtures count as their room."""
    label = fill_fixtures(label)
    out = {}
    for c in C16.group("S"):
        n, _, st, _ = cv2.connectedComponentsWithStats((label == c).astype(np.uint8), connectivity=4)
        if n > 1:
            out[c] = st[1:, cv2.CC_STAT_AREA].tolist()
    return out


SIDE_CLASSES = C16.group("S") + C16.group("OD") + C16.group("BG")


def door_side_pair(label, door, k):
    """The space types on the two faces of one door component (bool mask), or None.

    The door's longer bounding-box side gives its direction; the faces are probed k px
    away perpendicular to it, and each face takes its most common space type.
    """
    ys, xs = np.nonzero(door)
    if len(ys) == 0:
        return None
    h, w = label.shape
    horizontal = (xs.max() - xs.min()) >= (ys.max() - ys.min())
    sides = []
    for sgn in (-1, 1):
        y = np.clip(ys + sgn * k, 0, h - 1) if horizontal else ys
        x = xs if horizontal else np.clip(xs + sgn * k, 0, w - 1)
        v = label[y, x]
        v = v[np.isin(v, SIDE_CLASSES)]
        if len(v) == 0:
            return None
        sides.append(int(np.bincount(v).argmax()))
    return tuple(sorted(sides))


def door_pairs(label, k):
    """Space-type pairs joined by each door of the plan: [(a, b)] with class indices a <= b."""
    n, comp = components(label == C16.index["door"])
    pairs = [door_side_pair(label, comp == i, k) for i in range(1, n + 1)]
    return [p for p in pairs if p is not None]


def fill_fixtures(label, max_steps=200):
    """Copy of `label` in which every fixture pixel takes the room type of the room it stands in
    (user rule 2026-10-17: a room instance includes the fixtures standing in it).

    Rooms grow into fixture pixels one 4-neighbour step at a time, so the fill never crosses a
    wall (a straight-line nearest room could lie behind a 3-px wall). Fixtures touching no room
    keep their class.
    """
    fix = np.isin(label, C16.group("F"))
    if not fix.any() or not np.isin(label, C16.group("S")).any():
        return label
    out = label.copy()
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    for _ in range(max_steps):
        todo = np.isin(out, C16.group("F"))
        if not todo.any():
            break
        changed = False
        for c in C16.group("S"):
            grow = (cv2.dilate((out == c).astype(np.uint8), cross) > 0) & todo
            if grow.any():
                out[grow] = c
                todo &= ~grow
                changed = True
        if not changed:
            break
    return out
