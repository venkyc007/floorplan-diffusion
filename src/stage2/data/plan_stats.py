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
