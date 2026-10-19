"""R6 "windowless exterior span" (user, 2026-10-19): a habitable room with a long run of exterior wall and no
window in it -> "window missing" in the middle of that run.

    python -m src.stage2.detect.window_runs --zip completion_train.zip   -> configs/stage2/window_rule.yaml

Exterior run (completion scale, walls 3 px): the room's own boundary pixels that face the outside THROUGH WALL ONLY:
looking straight up / down / left / right from the room pixel, 1..(T_wall + 3) px of wall and then outdoor or
background (a window or door on the way does not count, so windows and doors break the runs). Runs = connected
lines of such pixels per direction; length = their extent along the wall.
Habitable rooms: living, bedroom, kitchen, and other_room instances >= need_area px (the R5 cut-off).
Thresholds (data-driven): the 95th percentile of each room type's LONGEST windowless exterior run over the
TRAINING ground-truth plans (rooms with any exterior wall). On a parsed plan a run longer than its room type's
threshold x `scale` (tuned) gives a "window" candidate: the box = a median window length centred on the run's
middle, across the wall.
"""

import argparse

import cv2
import numpy as np
import yaml

from src.stage2.constraints.constraints import CONST
from src.stage2.data.classes16 import C16
from src.stage2.detect.rules import Candidate, box_of
from src.utils import PROJECT_ROOT

I = C16.index
OUT_FILE = PROJECT_ROOT / "configs" / "stage2" / "window_rule.yaml"
HABITABLE = ["living", "bedroom", "kitchen", "other_room"]
DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def facing(label, depth=None):
    """{direction: (H, W) bool} pixels whose straight look in that direction crosses only wall and reaches OUT."""
    depth = depth or CONST["T_wall"] + 3
    h, w = label.shape
    wall = label == I["wall"]
    out_ = np.isin(label, C16.group("OUT"))
    res = {}
    for name, (dy, dx) in DIRS.items():
        alive = np.ones((h, w), bool)
        reach = np.zeros((h, w), bool)
        for k in range(1, depth + 1):
            sh = np.zeros((h, w), bool)
            so = np.zeros((h, w), bool)
            ys0, ys1 = max(0, -dy * k), h - max(0, dy * k)
            xs0, xs1 = max(0, -dx * k), w - max(0, dx * k)
            sh[ys0:ys1, xs0:xs1] = wall[ys0 + dy * k:ys1 + dy * k, xs0 + dx * k:xs1 + dx * k]
            so[ys0:ys1, xs0:xs1] = out_[ys0 + dy * k:ys1 + dy * k, xs0 + dx * k:xs1 + dx * k]
            if k > 1:
                reach |= alive & so
            alive &= sh
        res[name] = reach
    return res


def room_masks(label, need_area):
    out = []
    for name in HABITABLE:
        n, comp, stats, _ = cv2.connectedComponentsWithStats((label == I[name]).astype(np.uint8), connectivity=4)
        for i in range(1, n):
            if name == "other_room" and stats[i, cv2.CC_STAT_AREA] < need_area:
                continue
            out.append((name, comp == i))
    return out


def runs(label, need_area, face=None):
    """[(room type, direction, run mask)] of windowless exterior runs of every habitable room."""
    face = face or facing(label)
    out = []
    for name, m in room_masks(label, need_area):
        edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        for d, f in face.items():
            r = edge & f
            if not r.any():
                continue
            n, comp = cv2.connectedComponents(r.astype(np.uint8), connectivity=8)
            for i in range(1, n):
                out.append((name, d, comp == i))
    return out


def length(mask, d):
    ys, xs = np.nonzero(mask)
    return int((xs.max() - xs.min() + 1) if d in ("up", "down") else (ys.max() - ys.min() + 1))


def window_candidates(label, thresholds, need_area, win_len, scale=1.0):
    """[(Candidate("window", box, ["R6"], ratio))] for runs longer than threshold x scale; score = length / threshold."""
    out = []
    for name, d, m in runs(label, need_area):
        th = thresholds.get(name)
        L = length(m, d)
        if not th or L < th * scale:
            continue
        ys, xs = np.nonzero(m)
        cy, cx = int(np.median(ys)), int(np.median(xs))
        dy, dx = DIRS[d]
        box = np.zeros(label.shape, bool)
        k = CONST["T_wall"] + 3
        if d in ("up", "down"):
            y0, y1 = sorted((cy, cy + dy * k))
            box[max(0, y0):y1 + 1, max(0, cx - win_len // 2):cx + win_len // 2 + 1] = True
        else:
            x0, x1 = sorted((cx, cx + dx * k))
            box[max(0, cy - win_len // 2):cy + win_len // 2 + 1, max(0, x0):x1 + 1] = True
        out.append(Candidate("window", box_of(box, label.shape), ["R6"], L / th))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, help="completion_train.zip (ground-truth TRAINING plans)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--q", type=float, default=95)
    args = ap.parse_args()
    import zipfile
    need = yaml.safe_load((PROJECT_ROOT / "configs" / "stage2" / "detector_threshold.yaml").read_text(encoding="utf-8"))
    need = need.get("gap_rule", {}).get("need_area", 1514)
    longest = {n: [] for n in HABITABLE}
    z = zipfile.ZipFile(args.zip)                     # streamed one plan at a time (laptop RAM)
    names = sorted(n for n in z.namelist() if n.endswith(".png"))[: args.limit]
    for i, n in enumerate(names):
        lab = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_UNCHANGED)
        if (lab == 255).any():
            continue
        best = {}
        face = facing(lab)
        for name, m in room_masks(lab, need):
            Ls = []
            edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            for d, f in face.items():
                r = edge & f
                if not r.any():
                    continue
                n, comp = cv2.connectedComponents(r.astype(np.uint8), connectivity=8)
                Ls += [length(comp == j, d) for j in range(1, n)]
            if Ls:
                longest[name].append(max(Ls))
        if (i + 1) % 500 == 0:
            print(f"{i + 1} plans", flush=True)
    th = {n: float(np.percentile(v, args.q)) for n, v in longest.items() if v}
    out = {"source": f"{args.q:.0f}th percentile of each room type's longest windowless exterior run, completion_train "
                     "ground truth (src/stage2/detect/window_runs.py)",
           "need_area": need, "thresholds": {k: round(v, 1) for k, v in th.items()},
           "median_longest": {n: float(np.median(v)) for n, v in longest.items() if v},
           "rooms": {n: len(v) for n, v in longest.items()}}
    OUT_FILE.write_text("# GENERATED by python -m src.stage2.detect.window_runs\n" + yaml.safe_dump(out, sort_keys=False),
                        encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
