"""Architectural validity metrics (Day 6).

Pixel IoU says how much of the answer is right; these say whether the
result is a PLAUSIBLE floor plan. Every rule looks only at things that touch
the mask region grown by MARGIN (3) px, and is always reported next to the
ground truth's own count on the same case: real plans break some rules too
(e.g. free-standing wall stubs at open passages), so only an EXCESS over the
ground truth is a real defect.

Rules (counts per case):
  1 dead_ends     wall end-points (of the wall's centre line, ignoring tiny
                  skeleton spurs) with no door, window, outdoor or background
                  within END_RADIUS px
  2 fragments     wall/door/window pieces < FRAGMENT_PX px not connected to the
                  main boundary (the largest boundary piece)
  3 opening_fit   doors/windows that fail any of: wall at both ends along the
                  long axis ("bridges"), thickness within +-1 px of the narrower
                  end wall ("thickness"), end walls covering the opening's
                  thickness span with at least one matching it ("aligned");
                  tolerant of openings next to corners (common in real plans)
  4 door_spaces   doors without room/outdoor/background on both sides across the
                  door (background = outside: real front doors open onto it)
  5 window_out    windows without outdoor/background on either side
  6 unreachable   rooms (enclosed regions) not connected through doors to the
                  rest of the plan
  7 slivers       room components below the 1st-percentile area or width of
                  real training rooms (thresholds: configs/arch_metrics.yaml)
  8 rectangularity  room area / bounding-box area (mean; compared with ground truth)
Overall validity: a case is valid if, for rules 1-6 (rule 3 = its "bridges"
check only), the prediction has no more violations than the ground truth has
on the same case.

Main entry points:
  case_metrics(pred, gt, mask, thresholds)  -> dict of counts (pred and gt)
  summarize_arch(list of dicts)             -> rates per case + validity
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize

from src.data.classes import BACKGROUND, DOOR, OUTDOOR, ROOM, WALL, WINDOW

MARGIN = 3
END_RADIUS = 3
SPUR_PX = 3
FRAGMENT_PX = 10
# Rules that count towards the overall validity score. Opening fit enters
# only through its "bridges" check: real plans fail the thickness/alignment
# checks too often (28% of 500 real plans) for them to signal a defect;
# they are reported as diagnostics instead.
RULES = ["dead_ends", "fragments", "opening_bridge", "door_spaces", "window_out", "unreachable"]
EIGHT = np.ones((3, 3), np.uint8)


# ------------------------------------------------------------ helpers

def grow(mask, px):
    return cv2.dilate(mask.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)).astype(bool)


def components(binary, connectivity=4):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=connectivity)
    return n, lab, stats


def window_of(img, r, c, rad):
    """The square patch of radius rad around (r, c), clipped to the image."""
    return img[max(0, r - rad):r + rad + 1, max(0, c - rad):c + rad + 1]


# ---------------------------------------------------- 1. dead-end walls

def skeleton_endpoints(skel):
    """Skeleton pixels with exactly one 8-neighbour, minus short spurs
    (endpoints that reach a junction within SPUR_PX steps)."""
    nb = cv2.filter2D(skel.astype(np.uint8), -1, EIGHT, borderType=cv2.BORDER_CONSTANT) - skel
    ends = np.argwhere(skel & (nb == 1))
    junction = skel & (nb >= 3)
    keep = []
    for r, c in ends:
        prev, cur, spur = None, (r, c), False
        for _ in range(SPUR_PX):
            rr, cc = cur
            nbrs = [(rr + dr, cc + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                    if (dr or dc) and 0 <= rr + dr < skel.shape[0] and 0 <= cc + dc < skel.shape[1]
                    and skel[rr + dr, cc + dc] and (rr + dr, cc + dc) != prev]
            if not nbrs:
                break
            prev, cur = cur, nbrs[0]
            if junction[cur]:
                spur = True
                break
        if not spur:
            keep.append((r, c))
    return keep


def dead_ends(label, region):
    skel = skeletonize(label == WALL)
    n = 0
    for r, c in skeleton_endpoints(skel):
        if region[r, c] and not np.isin(window_of(label, r, c, END_RADIUS),
                                        [DOOR, WINDOW, OUTDOOR, BACKGROUND]).any():
            n += 1
    return n


# --------------------------------------------------- 2. floating fragments

def fragments(label, region):
    boundary = np.isin(label, [WALL, DOOR, WINDOW])
    n, lab, stats = components(boundary, connectivity=8)
    if n <= 2:
        return 0
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return sum(1 for i in range(1, n) if i != main and stats[i, cv2.CC_STAT_AREA] < FRAGMENT_PX
               and region[lab == i].any())


# --------------------------------------------- 3-5. openings (fit, doors, windows)

def opening_geometry(comp):
    """Bounding box and whether the opening runs horizontally (long axis along columns)."""
    rows, cols = np.nonzero(comp)
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    return r0, r1, c0, c1, (c1 - c0) >= (r1 - r0)


PROBE_PX = 4


def wall_run_at(label, r0, r1, c0, c1, horizontal, offset):
    """Wall extent (start, end) across the thickness direction, `offset` px
    beyond one end of the opening (negative offset = before the start)."""
    H, W = label.shape
    if horizontal:
        col = c0 + offset if offset < 0 else c1 + offset
        if not 0 <= col < W:
            return None
        idx = np.nonzero(label[max(0, r0 - 2):min(H, r1 + 3), col] == WALL)[0] + max(0, r0 - 2)
    else:
        row = r0 + offset if offset < 0 else r1 + offset
        if not 0 <= row < H:
            return None
        idx = np.nonzero(label[row, max(0, c0 - 2):min(W, c1 + 3)] == WALL)[0] + max(0, c0 - 2)
    return (int(idx.min()), int(idx.max())) if len(idx) else None


def end_wall_runs(label, r0, r1, c0, c1, horizontal):
    """For each end of the opening: the NARROWEST wall cross-section found
    1..PROBE_PX px beyond it, or None if there is no wall there.

    Probing only the pixel right next to the end is wrong near corners and
    T-junctions: it crosses the perpendicular wall, which looks 2-4 px too
    thick and offset (found on real plans). The narrowest cross-section is
    the straight wall that continues past the opening."""
    runs = []
    for side in (-1, 1):
        found = [wall_run_at(label, r0, r1, c0, c1, horizontal, side * d) for d in range(1, PROBE_PX + 1)]
        found = [f for f in found if f is not None]
        runs.append(min(found, key=lambda f: f[1] - f[0]) if found else None)
    return runs


def opening_checks(label, comp):
    """(bridges, thickness_ok, aligned) for one door/window component."""
    r0, r1, c0, c1, horizontal = opening_geometry(comp)
    lo, hi = (r0, r1) if horizontal else (c0, c1)            # extent across the thickness
    runs = end_wall_runs(label, r0, r1, c0, c1, horizontal)
    bridges = all(run is not None for run in runs)
    if not bridges:
        return False, False, False
    thick = hi - lo + 1
    # Openings often sit right next to a corner, where the wall beyond that end
    # is the (wider-looking) perpendicular wall. So: thickness is compared with
    # the NARROWER end wall, and alignment needs both end walls to cover the
    # opening's thickness span with at least one matching it exactly.
    thickness_ok = abs(min(b - a + 1 for a, b in runs) - thick) <= 1
    covers = all(a <= lo + 1 and b >= hi - 1 for a, b in runs)
    exact = any(abs(a - lo) <= 1 and abs(b - hi) <= 1 for a, b in runs)
    return bridges, thickness_ok, covers and exact


def sides_across(label, comp):
    """Classes of the pixels just beyond the opening on both sides across its thickness."""
    r0, r1, c0, c1, horizontal = opening_geometry(comp)
    H, W = label.shape
    if horizontal:
        a = label[r0 - 1, c0:c1 + 1] if r0 > 0 else np.array([BACKGROUND])
        b = label[r1 + 1, c0:c1 + 1] if r1 < H - 1 else np.array([BACKGROUND])
    else:
        a = label[r0:r1 + 1, c0 - 1] if c0 > 0 else np.array([BACKGROUND])
        b = label[r0:r1 + 1, c1 + 1] if c1 < W - 1 else np.array([BACKGROUND])
    return a, b


def majority_in(values, classes):
    return np.isin(values, classes).mean() > 0.5


def openings(label, region):
    """Counts for rules 3-5 plus the per-check pass numbers of rule 3."""
    out = {"opening_fit": 0, "opening_bridge": 0, "door_spaces": 0, "window_out": 0,
           "n_openings": 0, "bridges_ok": 0, "thickness_ok": 0, "aligned_ok": 0}
    for cls in (DOOR, WINDOW):
        n, lab, _ = components(label == cls)
        for i in range(1, n):
            comp = lab == i
            if not region[comp].any():
                continue
            out["n_openings"] += 1
            b, t, a = opening_checks(label, comp)
            out["bridges_ok"] += b
            out["thickness_ok"] += t
            out["aligned_ok"] += a
            out["opening_fit"] += not (b and t and a)
            out["opening_bridge"] += not b
            s1, s2 = sides_across(label, comp)
            # Background counts as an outside space: real front doors open onto it.
            if cls == DOOR and not (majority_in(s1, [ROOM, OUTDOOR, BACKGROUND])
                                    and majority_in(s2, [ROOM, OUTDOOR, BACKGROUND])):
                out["door_spaces"] += 1
            if cls == WINDOW and not (majority_in(s1, [OUTDOOR, BACKGROUND]) or majority_in(s2, [OUTDOOR, BACKGROUND])):
                out["window_out"] += 1
    return out


# ---------------------------------------------------- 6. room reachability

def unreachable_rooms(label, region):
    """Number of unreachable rooms (see unreachable_room_masks)."""
    return len(unreachable_room_masks(label, region))


def unreachable_room_masks(label, region):
    """Rooms (enclosed regions) not connected through doors to the rest of the plan.

    Nodes: enclosed regions (non-boundary areas not touching the image edge)
    plus one 'outside' node (the edge-touching areas). Open gaps in walls
    already merge regions. Edges: a door touching two different nodes.
    Returned: one True/False mask per region outside the main group that touches the region."""
    open_space = ~np.isin(label, [WALL, DOOR, WINDOW])
    n, lab, _ = components(open_space, connectivity=4)
    edge_ids = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    node = np.arange(n)
    for e in edge_ids:
        node[e] = -1                                        # all edge-touching areas = 'outside'
    parent = {k: k for k in set(node[1:])}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    nd, dlab, _ = components(label == DOOR)
    for i in range(1, nd):
        ring = cv2.dilate((dlab == i).astype(np.uint8), EIGHT, iterations=2).astype(bool)
        touched = {node[v] for v in np.unique(lab[ring]) if v > 0}
        touched = list(touched)
        for a, b in zip(touched, touched[1:]):
            parent[find(a)] = find(b)
    rooms = [k for k in range(1, n) if k not in edge_ids]
    if not rooms:
        return []
    groups = {}
    for k in rooms:
        groups.setdefault(find(node[k]), []).append(k)
    outside_group = find(-1) if -1 in parent else None
    # the "main" group: the one containing the outside, else the largest
    main = outside_group if outside_group in groups else max(groups, key=lambda g: len(groups[g]))
    return [lab == k for k in rooms if find(node[k]) != main and region[lab == k].any()]


# ------------------------------------------------ 7-8. slivers, rectangularity

def room_shapes(label, region):
    """(area, width, rectangularity) of room components touching the region.
    width = 2 x largest distance to a non-room pixel (inscribed width)."""
    n, lab, stats = components(label == ROOM)
    dist = cv2.distanceTransform((label == ROOM).astype(np.uint8), cv2.DIST_L2, 3)
    shapes = []
    for i in range(1, n):
        comp = lab == i
        if not region[comp].any():
            continue
        area = int(stats[i, cv2.CC_STAT_AREA])
        box = stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT]
        shapes.append((area, 2 * float(dist[comp].max()), area / box))
    return shapes


def slivers(shapes, thresholds):
    return sum(1 for a, w, _ in shapes if a < thresholds["min_area"] or w < thresholds["min_width"])


# ------------------------------------------------------------ per case

def label_metrics(label, region, thresholds):
    shapes = room_shapes(label, region)
    out = {"dead_ends": dead_ends(label, region), "fragments": fragments(label, region),
           "unreachable": unreachable_rooms(label, region), "slivers": slivers(shapes, thresholds),
           "rect_sum": sum(r for _, _, r in shapes), "n_rooms": len(shapes)}
    out.update(openings(label, region))
    return out


def case_metrics(pred, gt, mask, thresholds):
    """Counts for the prediction (composited with the known pixels) and the ground truth."""
    region = grow(mask, MARGIN)
    pred = np.where(mask, pred, gt)
    p, g = label_metrics(pred, region, thresholds), label_metrics(gt, region, thresholds)
    out = {f"pred_{k}": v for k, v in p.items()}
    out.update({f"gt_{k}": v for k, v in g.items()})
    out["valid"] = all(p[r] <= g[r] for r in RULES)
    return out


def summarize_arch(rows):
    """Rates per case (prediction and ground truth), opening-fit pass rates,
    mean rectangularity and the overall validity share."""
    import pandas as pd
    d = pd.DataFrame(rows)
    t = d.sum(numeric_only=True)
    out = {"n": len(d)}
    for k in RULES + ["slivers"]:
        out[f"{k}_per_case"] = d[f"pred_{k}"].mean()
        out[f"gt_{k}_per_case"] = d[f"gt_{k}"].mean()
    for who in ("pred", "gt"):
        n_op = t[f"{who}_n_openings"]
        pre = "" if who == "pred" else "gt_"
        out[f"{pre}opening_fit_pct"] = 100 * (1 - t[f"{who}_opening_fit"] / n_op) if n_op else np.nan
        for chk in ("bridges", "thickness", "aligned"):
            out[f"{pre}{chk}_pct"] = 100 * t[f"{who}_{chk}_ok"] / n_op if n_op else np.nan
        out[f"{pre}rectangularity"] = t[f"{who}_rect_sum"] / t[f"{who}_n_rooms"] if t[f"{who}_n_rooms"] else np.nan
    out["arch_validity_pct"] = 100 * d["valid"].mean()
    return out


# ------------------------------------------------------ sliver thresholds

def compute_thresholds(labels):
    """1st-percentile room area and inscribed width over all room components
    of real (training) plans."""
    areas, widths = [], []
    everywhere = None
    for label in labels:
        if everywhere is None:
            everywhere = np.ones(label.shape, bool)
        for a, w, _ in room_shapes(label, everywhere):
            areas.append(a)
            widths.append(w)
    return {"min_area": float(np.percentile(areas, 1)), "min_width": float(np.percentile(widths, 1)),
            "n_rooms": len(areas)}


if __name__ == "__main__":
    # python -m src.eval.arch_metrics  -> writes configs/arch_metrics.yaml from the training plans
    import yaml
    from src.data.dataset import load_labels
    from src.utils import project_path
    th = compute_thresholds(load_labels(project_path("data/processed/train.npz")))
    text = ("# Sliver thresholds for src/eval/arch_metrics.py (rule 7), computed by\n"
            "#     python -m src.eval.arch_metrics\n"
            f"# from {th['n_rooms']} room components of the training plans: 1st percentiles.\n")
    text += yaml.safe_dump({"min_area": round(th["min_area"], 1), "min_width": round(th["min_width"], 2)})
    project_path("configs/arch_metrics.yaml").write_text(text, encoding="utf-8")
    print(text)


# ------------------------------------------------ locations (for figures)

def violation_locations(label, region, thresholds):
    """{rule: list of True/False pixel masks}, one mask per violation, for drawing.
    Uses the same rules as label_metrics."""
    out = {k: [] for k in RULES + ["slivers", "low_rectangularity"]}
    skel = skeletonize(label == WALL)
    for r, c in skeleton_endpoints(skel):
        if region[r, c] and not np.isin(window_of(label, r, c, END_RADIUS),
                                        [DOOR, WINDOW, OUTDOOR, BACKGROUND]).any():
            spot = np.zeros(label.shape, bool)
            spot[max(0, r - 2):r + 3, max(0, c - 2):c + 3] = True
            out["dead_ends"].append(spot)
    boundary = np.isin(label, [WALL, DOOR, WINDOW])
    n, lab, stats = components(boundary, connectivity=8)
    if n > 2:
        main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out["fragments"] += [lab == i for i in range(1, n) if i != main
                             and stats[i, cv2.CC_STAT_AREA] < FRAGMENT_PX and region[lab == i].any()]
    for cls in (DOOR, WINDOW):
        n, lab, _ = components(label == cls)
        for i in range(1, n):
            comp = lab == i
            if not region[comp].any():
                continue
            if not opening_checks(label, comp)[0]:
                out["opening_bridge"].append(comp)
            s1, s2 = sides_across(label, comp)
            if cls == DOOR and not (majority_in(s1, [ROOM, OUTDOOR, BACKGROUND])
                                    and majority_in(s2, [ROOM, OUTDOOR, BACKGROUND])):
                out["door_spaces"].append(comp)
            if cls == WINDOW and not (majority_in(s1, [OUTDOOR, BACKGROUND]) or majority_in(s2, [OUTDOOR, BACKGROUND])):
                out["window_out"].append(comp)
    out["unreachable"] = unreachable_room_masks(label, region)
    n, lab, stats = components(label == ROOM)
    dist = cv2.distanceTransform((label == ROOM).astype(np.uint8), cv2.DIST_L2, 3)
    for i in range(1, n):
        comp = lab == i
        if not region[comp].any():
            continue
        area = stats[i, cv2.CC_STAT_AREA]
        if area < thresholds["min_area"] or 2 * dist[comp].max() < thresholds["min_width"]:
            out["slivers"].append(comp)
        if area / (stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT]) < 0.6:
            out["low_rectangularity"].append(comp)
    return out
