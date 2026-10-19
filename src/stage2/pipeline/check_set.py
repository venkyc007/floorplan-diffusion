"""Visual check set (user, chunk 8): 12 val drawings with one known omission each, run end to end after
every fix from now on.

    python -m src.stage2.pipeline.check_set --zip parser_val.zip   -> outputs/stage2/check_set.png + .md

Cases (12): 2 door (a) "opening forgotten", 2 door (b) "gap left, no door", 2 window, 2 wall, 2 fixture,
1 missing room, 1 intact. Drawings are taken in file order (no cherry-picking): for each case kind, the
first typed val drawings not used yet that have such a component. The omission is made on the IMAGE:
  door (a)  a door whose removal cuts a room off (the TA case, caught by C8): its pixels are inked with the
            local wall grey, i.e. a line of wall thickness across the doorway
  door (b)  a door between two spaces that need one (the R5 pair rule): doorway left open
  window / fixture / wall   the component erased (a wall piece = 5 wall thicknesses of a straight run)
  room      the walls and doors between a room and its neighbour erased
Erasure (src/stage2/data/erasure.py): the component's own label pixels grown 2 px, inpainted from the
surroundings (Telea); door (a) = doorway painted in the local wall colour; door (b) = doorway + leaf + arc
inpainted. The orange outline is drawn on the figure axes only, never into the pipeline's input. The pipeline runs in fast mode.
Scoring (automatic, on the completion-scale map): found = a hole box overlaps the omission; restored = an
addition of the expected class overlaps it (door; window; wall; that fixture; wall for a room);
extra = added components (wall/door/window/railing/fixture) away from it.
Correct = restored and no extra; for the intact drawing, correct = nothing added.
"""

import argparse
import time
import zipfile

import cv2
import numpy as np

from src.stage2.constraints.rules import components, room_instances
from src.stage2.data.classes16 import C16
from src.stage2.data.erasure import draw_wall, erase, erase_symbol, open_gap
from src.stage2.data.plan_stats import door_side_pair
from src.stage2.detect import rules as R
from src.stage2.detect.gaps import closed_pairs
from src.stage2.detect.rules_diag import single_door_rooms
from src.stage2.pipeline.run import REAL, run_pipeline
from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "stage2"
I = C16.index
PLAN = [("door (a)", 2), ("door (b)", 2), ("window", 2), ("wall", 2), ("fixture", 2), ("room", 1), ("intact", 1)]
WALL_PX = 8                       # parser-set wall thickness


def read(z, name, flag):
    return cv2.imdecode(np.frombuffer(z.read(name), np.uint8), flag)


def grow(m, r):
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1,) * 2, np.uint8)) > 0


def door_a(lbl):
    for room, door in single_door_rooms(lbl):
        plan = lbl.copy()
        plan[door] = I["wall"]
        if room.sum() >= 200 and R.reachable_parts(plan).any() and not (R.reachable_parts(plan) & room).any():
            return door
    return None


def door_b(lbl, bad, min_len=0):
    """A door R5 is meant for: closed pair, one side a bedroom or bathroom, doorway >= min_len (image px)."""
    for d in components(lbl == I["door"]):
        pair = door_side_pair(lbl, d, 2 * WALL_PX)
        ys, xs = np.nonzero(d)
        length = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
        if (d.sum() >= 40 and pair and pair[0] != pair[1] and bad[pair[0], pair[1]] and length >= min_len
                and {C16.names[p] for p in pair} & {"bedroom", "bathroom"}):
            return d
    return None


def first_component(lbl, classes, min_px=40):
    comps = [m for m in components(np.isin(lbl, classes)) if m.sum() >= min_px]
    return comps[0] if comps else None


def wall_piece(lbl):
    """A piece of a straight wall run, 3-6 wall thicknesses long (first long run in raster order)."""
    wall = (lbl == I["wall"]).astype(np.uint8)
    for k, horizontal in ((np.ones((1, 8 * WALL_PX), np.uint8), True), (np.ones((8 * WALL_PX, 1), np.uint8), False)):
        runs = components(cv2.morphologyEx(wall, cv2.MORPH_OPEN, k) > 0)
        if not runs:
            continue
        m = runs[0]
        ys, xs = np.nonzero(m)
        piece = np.zeros_like(m)
        L = 5 * WALL_PX
        if horizontal:
            c = (xs.min() + xs.max()) // 2
            piece[:, c - L // 2:c + L // 2] = m[:, c - L // 2:c + L // 2]
        else:
            c = (ys.min() + ys.max()) // 2
            piece[c - L // 2:c + L // 2] = m[c - L // 2:c + L // 2]
        return piece
    return None


def room_walls(lbl):
    """Walls + doors between the first sizeable room and its main room neighbour."""
    r = 2 * WALL_PX
    for c, m in room_instances(lbl):
        if m.sum() < 2000:
            continue
        ring = grow(m, r) & ~m
        neigh = lbl[ring & np.isin(lbl, C16.group("S")) & (lbl != c)]
        if len(neigh) == 0:
            continue
        other = (lbl == int(np.bincount(neigh).argmax())) & ~m
        band = grow(m, r) & grow(other, r) & np.isin(lbl, [I["wall"], I["door"]])
        if band.sum() >= 100:
            return band
    return None


def make_case(kind, img, lbl, bad, min_len=0):
    """(image with the omission, removed mask in image px, expected added class) or None."""
    if kind == "intact":
        return img, np.zeros(lbl.shape, bool), None
    m = {"door (a)": lambda: door_a(lbl), "door (b)": lambda: door_b(lbl, bad, min_len),
         "window": lambda: first_component(lbl, [I["window"]]), "fixture": lambda: first_component(lbl, C16.group("F")),
         "wall": lambda: wall_piece(lbl), "room": lambda: room_walls(lbl)}[kind]()
    if m is None or not m.any():
        return None
    if kind == "door (a)":                               # wall drawn across the doorway, leaf + arc removed
        out = draw_wall(img, m, lbl, WALL_PX)
    elif kind == "door (b)":                             # doorway left open, leaf + arc removed
        out = open_gap(img, m, lbl, WALL_PX)
    elif kind in ("window", "fixture"):                  # whole symbol box cleared (window: wall drawn across)
        out = erase_symbol(img, m, lbl, WALL_PX)
    else:                                                # inpainted from the surroundings (never white)
        out = erase(img, m, lbl, wall_px=WALL_PX)
    expected = {"door (a)": "door", "door (b)": "door", "window": "window", "wall": "wall", "room": "wall",
                "fixture": C16.names[int(np.bincount(lbl[m]).argmax())]}[kind]
    return out, m, expected


def judge(res, removed, expected):
    comp = res["completed"]
    H, W = comp.shape
    rm = cv2.resize(removed.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    rm = grow(rm, 3) if rm.any() else rm
    found = any(rm[h["box"][0]:h["box"][2] + 1, h["box"][1]:h["box"][3] + 1].any() for h in res["holes"])
    added = (comp != res["parsed"]) & np.isin(comp, [I[c] for c in REAL])
    restored = bool(expected) and bool((added & (comp == I[expected]) & rm).any())
    extra = sum(1 for m in components(added) if not (m & rm).any())
    correct = (not added.any()) if expected is None else (restored and extra == 0)
    return {"found": found, "restored": restored, "extra": extra, "correct": correct}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: only the first N cases")
    args = ap.parse_args()
    import yaml
    g = yaml.safe_load((PROJECT_ROOT / "configs" / "stage2" / "detector_threshold.yaml").read_text(encoding="utf-8"))
    g = g.get("gap_rule", {})
    bad = closed_pairs(open_rate=g.get("open_rate", 0.10))                    # pairs R5 covers
    min_len = g.get("min_len", 0) * WALL_PX / 3                                 # completion px -> image px
    z = zipfile.ZipFile(args.zip)
    names = sorted(n[:-8] for n in z.namelist() if n.endswith("_img.png"))
    used, cases = set(), []
    for kind, count in PLAN:
        if args.limit and len(cases) >= args.limit:
            break
        got = 0
        for n in names:
            if got >= count:
                break
            if n in used:
                continue
            lbl = read(z, f"{n}_lbl.png", cv2.IMREAD_UNCHANGED)
            if (lbl == 255).any():                   # untyped plan: skip
                continue
            img = read(z, f"{n}_img.png", cv2.IMREAD_GRAYSCALE)
            c = make_case(kind, img, lbl, bad, min_len)
            if c is None:
                continue
            used.add(n)
            got += 1
            t = time.time()
            res = run_pipeline(c[0], fast=True)
            res["seconds"] = time.time() - t
            verdict = judge(res, c[1], c[2])
            cases.append((n, kind, c, res, verdict))
            print(n, kind, f"{res['seconds']:.0f} s", verdict, res["messages"], flush=True)
    figure(cases, args.tag)


def figure(cases, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(len(cases), 3, figsize=(18, 6 * len(cases)))
    score = sum(v["correct"] for *_, v in cases)
    L = [f"# Check set{tag}: {score} / {len(cases)} correct (fast mode, CPU)", "",
         "found = a hole overlaps the omission; restored = the expected class was added there; extra = added "
         "components elsewhere; correct = restored and no extra (intact: nothing added).", "",
         "| # | drawing | omission | found | restored | extra | correct | holes | s |", "|---|---|---|---|---|---|---|---|---|"]
    for i, (n, kind, (white, removed, expected), res, v) in enumerate(cases):
        L.append(f"| {i + 1} | {n} | {kind} | {'yes' if v['found'] else 'no'} | {'yes' if v['restored'] else 'no'} | "
                 f"{v['extra']} | {'**yes**' if v['correct'] else 'no'} | {len(res['holes'])} | {res['seconds']:.0f} |")
    for i, (n, kind, (white, removed, expected), res, v) in enumerate(cases):
        a, b, c = axes[i]
        a.imshow(white, cmap="gray", vmin=0, vmax=255)
        if removed.any():
            ys, xs = np.nonzero(removed)
            a.add_patch(Rectangle((xs.min() - 4, ys.min() - 4), xs.max() - xs.min() + 8, ys.max() - ys.min() + 8,
                                  fill=False, ec="orange", lw=1.5))
        a.set_title(f"{i + 1}. {n[-28:]}: {kind} removed (orange)", fontsize=9)
        b.imshow(C16.colourise(res["parsed"]), interpolation="nearest")
        for h in res["holes"]:
            hy0, hx0, hy1, hx1 = h["box"]
            b.add_patch(Rectangle((hx0, hy0), hx1 - hx0, hy1 - hy0, fill=False, ec="red", lw=1.2))
            b.text(hx0, hy0 - 2, f"{h['type']} ({'+'.join(h['rules'])})", color="red", fontsize=7)
        b.set_title(f"parsed + detected holes ({len(res['holes'])})", fontsize=9)
        c.imshow(res["overlay"])
        c.set_title(f"result: {'CORRECT' if v['correct'] else 'wrong'}", fontsize=9)
        c.text(0, res["overlay"].shape[0] * 1.02, "\n".join(res["messages"][:6]), fontsize=8, va="top")
    L.append("")
    for i, (n, kind, _, res, v) in enumerate(cases):
        L += [f"## {i + 1}. {n} ({kind}): automatic verdict {'correct' if v['correct'] else 'wrong'}", "",
              f"Removed: {kind}. Detected: " + (", ".join(f"{h['type']} ({'+'.join(h['rules'])})" for h in res["holes"])
                                                 or "nothing") +
              f". Found {'yes' if v['found'] else 'no'}, restored {'yes' if v['restored'] else 'no'}, extra {v['extra']}.",
              "", "Added (messages):", ""]
        L += [f"- {m}" for m in res["messages"]] or ["- (nothing)"]
        L.append("")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / f"check_set{tag}.png", dpi=60)
    (OUT / f"check_set{tag}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:20]))


if __name__ == "__main__":
    main()
