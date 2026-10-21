"""Visual check set (user, chunk 8): 12 val drawings with one known omission each, run end to end after
every fix from now on.

    python -m src.stage2.pipeline.check_set --zip parser_val.zip   -> outputs/stage2/history/check_set.png + .md

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
New definition (user, 2026-10-20; e2e_val.score): fixed = a flag there AND the final plan has the right class at the
omission with IoU >= 0.3, whoever drew it (parser symbols reported "missing in the drawing" count); false adds =
added or marked components away from it; correct (new) = fixed and no false adds (intact: nothing added or marked).
"""

import argparse
import time
import zipfile

import cv2
import numpy as np

from src.stage2.constraints.rules import components, room_instances
from src.stage2.data.classes16 import C16
from src.stage2.data.erasure import draw_wall, erase, erase_symbol, open_gap
from src.stage2.data.omission_meta import omission_meta
from src.stage2.data.plan_stats import door_side_pair
from src.stage2.detect import rules as R
from src.stage2.detect.gaps import closed_pairs
from src.stage2.detect.rules_diag import single_door_rooms
from src.stage2.pipeline.e2e_val import score
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
    """(image with the omission, removed mask in image px, expected added class, metadata) or None.
    metadata: type / location / position / size bin (src/stage2/data/omission_meta.py)."""
    if kind == "intact":
        return img, np.zeros(lbl.shape, bool), None, omission_meta(lbl, None, "intact")
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
    return out, m, expected, omission_meta(lbl, m, kind)


def judge(res, removed, expected, kind=None, lbl=None):
    comp = res["completed"]
    H, W = comp.shape
    rm = cv2.resize(removed.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    rm = grow(rm, 3) if rm.any() else rm
    found = any(rm[h["box"][0]:h["box"][2] + 1, h["box"][1]:h["box"][3] + 1].any() for h in res["holes"])
    added = (comp != res["parsed"]) & np.isin(comp, [I[c] for c in REAL])
    restored = bool(expected) and bool((added & (comp == I[expected]) & rm).any())
    extra = sum(1 for m in components(added) if not (m & rm).any())
    correct = (not added.any()) if expected is None else (restored and extra == 0)
    v = {"found": found, "restored": restored, "extra": extra, "correct": correct}
    if expected is None:
        marked = res.get("marked", np.zeros(comp.shape, bool))
        v.update(fixed=False, fix_iou=0.0, false_adds=len(components(added | marked)),
                 correct_new=not (added.any() or marked.any()))
    elif lbl is not None:
        s = score(res, removed, lbl, "room" if kind == "room" else expected)
        v.update(fixed=s["fixed"], fix_iou=s["fix_iou"], false_adds=s["false_adds"],
                 correct_new=s["fixed"] and s["false_adds"] == 0)
    return v


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
            verdict = judge(res, c[1], c[2], kind, lbl)
            cases.append((n, kind, c, res, verdict))
            print(n, kind, f"{res['seconds']:.0f} s", verdict, res["messages"], flush=True)
    figure(cases, args.tag)
    hole_report(cases, args.tag)


def hole_report(cases, tag):
    """Step 3 (user, 2026-10-19): for every hole that overlaps the omission, the map before the fill, the hybrid's
    RAW output (before the type restriction) and what the restriction kept -> check_set_holes{tag}.png and a
    section appended to check_set{tag}.md with the counts that explain the outcome. Nothing is changed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, L = [], ["", "## Holes in the right place: raw hybrid output (step 3)", "",
                   "Per hole overlapping the omission: new pixels of the hole's type in the RAW hybrid output (before "
                   "the type restriction), its largest new component, what the restriction kept, and the classes the "
                   "hybrid changed inside the box. Figure: check_set_holes" + tag + ".png.", "",
                   "| # | drawing | omission | hole | raw new px of type (largest) | kept px | hybrid changed (class: px) | why nothing / what |",
                   "|---|---|---|---|---|---|---|---|"]
    for i, (n, kind, (img, removed, expected, *_), res, v) in enumerate(cases):
        comp = res["completed"]
        rm = cv2.resize(removed.astype(np.uint8), (comp.shape[1], comp.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        rm = grow(rm, 3) if rm.any() else rm
        for h in res["holes"]:
            y0, x0, y1, x1 = h["box"]
            if not rm[y0:y1 + 1, x0:x1 + 1].any() or "raw" not in h:
                continue
            r = h["raw"]
            oy, ox = r["origin"]
            box = np.zeros(r["before"].shape, bool)
            box[y0 - oy:y1 - oy + 1, x0 - ox:x1 - ox + 1] = True
            changed = box & (r["hybrid"] != r["before"])
            typ = h["type"]
            cls = [I[typ]] if typ in I else ([I["wall"]] if typ == "room" else [])
            new = changed & np.isin(r["hybrid"], cls)
            comps = components(new)
            big = max((c.sum() for c in comps), default=0)
            kept = int((box & (r["kept"] != r["before"])).sum())
            cnt = {C16.names[c]: int((changed & (r["hybrid"] == c)).sum()) for c in np.unique(r["hybrid"][changed])}
            cnt = ", ".join(f"{k}: {v}" for k, v in sorted(cnt.items(), key=lambda t: -t[1])) or "none"
            if not changed.any():
                why = "the hybrid reproduced the map unchanged inside the box"
            elif not new.any():
                why = f"the hybrid changed pixels but drew no {typ}; the restriction reverts them all"
            elif kept == 0:
                why = (f"{typ} drawn but dropped: C15 (not a compatible room)" if typ in R.CONST.get("fixture_rooms", {})
                       else f"{typ} drawn but nothing kept")
            elif not v["restored"]:
                why = f"{typ} kept ({kept} px) but not on the omitted spot"
            else:
                why = f"{typ} kept ({kept} px) on the omitted spot"
            L.append(f"| {i + 1} | {n} | {kind} | {typ} ({'+'.join(h['rules'])}) | {int(new.sum())} ({big}) | {kept} | "
                     f"{cnt} | {why} |")
            rows.append((f"{i + 1}. {n}: {kind}, hole {typ}", r, box, rm[oy:oy + box.shape[0], ox:ox + box.shape[1]]))
    if not rows:
        L.append("| - | - | - | no hole overlapped an omission | | | | |")
    else:
        fig, axes = plt.subplots(len(rows), 3, figsize=(12, 4 * len(rows)), squeeze=False)
        for (title, r, box, rmc), ax in zip(rows, axes):
            for a, key, t in zip(ax, ("before", "hybrid", "kept"),
                                 ("parse (before fill)", "hybrid RAW output", "after type restriction")):
                a.imshow(C16.colourise(r[key]), interpolation="nearest")
                ys, xs = np.nonzero(box)
                a.add_patch(plt.Rectangle((xs.min() - 0.5, ys.min() - 0.5), xs.max() - xs.min() + 1,
                                          ys.max() - ys.min() + 1, fill=False, ec="red", lw=1))
                if rmc.any():
                    a.contour(rmc, levels=[0.5], colors="orange", linewidths=0.8)
                a.set_title(f"{title}: {t}" if key == "before" else t, fontsize=8)
                a.axis("off")
        fig.tight_layout()
        fig.savefig(OUT / f"check_set_holes{tag}.png", dpi=80)
    with open(OUT / f"check_set{tag}.md", "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def figure(cases, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(len(cases), 3, figsize=(18, 6 * len(cases)))
    n_ok = sum(v["correct"] for *_, v in cases)
    n_new = sum(v.get("correct_new", False) for *_, v in cases)
    L = [f"# Check set{tag}: {n_new} / {len(cases)} correct (new definition), {n_ok} / {len(cases)} (old) "
         "(fast mode, CPU)", "",
         "Old: found = a hole overlaps the omission; restored = the expected class was added there; extra = added "
         "components elsewhere; correct = restored and no extra (intact: nothing added). New (user, 2026-10-20): "
         "fixed = a flag there and the final plan has the right class at the omission with IoU >= 0.3, whoever "
         "drew it; false adds = added or marked ('symbol missing in the drawing') components elsewhere; correct = "
         "fixed and no false adds.", "",
         "| # | drawing | omission | found | restored | extra | correct (old) | fixed (IoU) | false adds | "
         "correct (new) | holes | s |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    yn = lambda b: "yes" if b else "no"
    for i, (n, kind, (white, removed, expected, *_), res, v) in enumerate(cases):
        L.append(f"| {i + 1} | {n} | {kind} | {yn(v['found'])} | {yn(v['restored'])} | {v['extra']} | "
                 f"{'**yes**' if v['correct'] else 'no'} | {yn(v.get('fixed'))} ({v.get('fix_iou', 0):.2f}) | "
                 f"{v.get('false_adds', '-')} | {'**yes**' if v.get('correct_new') else 'no'} | {len(res['holes'])} | "
                 f"{res['seconds']:.0f} |")
    for i, (n, kind, (white, removed, expected, *_), res, v) in enumerate(cases):
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
