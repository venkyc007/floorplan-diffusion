"""Held-out detection on DRAWINGS (image level, realistic erasure): door kinds (a) / (b) separately.

    python -m src.stage2.detect.image_eval --completion completion_val.zip --drawings parser_val.zip [--limit N]
        -> appends a section to outputs/stage2/detection_eval.md, writes outputs/stage2/image_eval_cases.csv

Plans: the 159 held-out val plans (not used for any setting; detection_eval.build_heldout_cases). The first 59
stay intact; each of the other 100 gets two separate omissions, kinds in turn over
door (a), door (b), window, wall, railing, fixture, room (seed 5; a kind the plan lacks -> the next kind).
Each omission is made on the drawing with src/stage2/data/erasure.py, then the pipeline's PARSE and CHECK
run (no fill). Methods: detector (0.9 / 40 px) alone, + C8, + C8 + R5 (the final CHECK).
Found = a proposal box (completion scale) overlaps the omitted pixels (resized, grown 3 px); in brackets:
that proposal also has the right type. False alarms = proposals on the 59 intact drawings, per drawing.
Door (a) is also shown on the subset where the doorway closure cuts a room off (C8's scope), door (b) on
the subset R5 is meant for (closed pair with a bedroom or bathroom side).
"""

import argparse
import csv
import time
import zipfile

import cv2
import numpy as np
import yaml

from src.stage2.constraints.rules import components, room_instances
from src.stage2.data.classes16 import C16
from src.stage2.data.erasure import draw_wall, erase, erase_symbol, grown, open_gap
from src.stage2.data.erasure_large import erase_region, erase_room
from src.stage2.data.omission_meta import FIELDS, breakdowns, omission_meta
from src.stage2.data.plan_stats import door_side_pair
from src.stage2.detect import rules as R
from src.stage2.detect.detection_eval import build_heldout_cases
from src.stage2.detect.gaps import closed_pairs
from src.stage2.pipeline.run import cfg, detect_parts, parse
from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "stage2"
I = C16.index
KINDS = ["door (a)", "door (b)", "window", "wall", "railing", "fixture", "room"]
TYPE = {"door (a)": "door", "door (b)": "door", "window": "window", "wall": "wall", "railing": "railing",
        "room": "room"}
WALL_PX = 8
TITLE = "final CHECK + window / wall ink double checks, re-tuned at budget 0.4 (2026-10-20)"


def read(z, name, flag):
    return cv2.imdecode(np.frombuffer(z.read(name), np.uint8), flag)


def wall_piece(lbl, rng):
    """A random piece of a straight wall run, 5 wall thicknesses long."""
    wall = (lbl == I["wall"]).astype(np.uint8)
    L = 5 * WALL_PX
    horizontal = rng.random() < 0.5
    k = np.ones((1, 8 * WALL_PX) if horizontal else (8 * WALL_PX, 1), np.uint8)
    runs = components(cv2.morphologyEx(wall, cv2.MORPH_OPEN, k) > 0)
    if not runs:
        return None
    m = runs[rng.integers(len(runs))]
    ys, xs = np.nonzero(m)
    piece = np.zeros_like(m)
    if horizontal:
        s = int(rng.integers(xs.min(), max(xs.min() + 1, xs.max() - L + 1)))
        piece[:, s:s + L] = m[:, s:s + L]
    else:
        s = int(rng.integers(ys.min(), max(ys.min() + 1, ys.max() - L + 1)))
        piece[s:s + L] = m[s:s + L]
    return piece if piece.any() else None


def room_walls(lbl, rng):
    """Walls + doors between a random sizeable room and its main room neighbour."""
    r = 2 * WALL_PX
    rooms = [(c, m) for c, m in room_instances(lbl) if m.sum() >= 2000]
    for i in rng.permutation(len(rooms)):
        c, m = rooms[i]
        ring = grown(m, r) & ~m
        neigh = lbl[ring & np.isin(lbl, C16.group("S")) & (lbl != c)]
        if len(neigh) == 0:
            continue
        other = (lbl == int(np.bincount(neigh).argmax())) & ~m
        band = grown(m, r) & grown(other, r) & np.isin(lbl, [I["wall"], I["door"]])
        if band.sum() >= 100:
            return band
    return None


LARGE_KINDS = ["room (whole)", "region"]        # src/stage2/data/erasure_large.py (user, 2026-10-20)


def omit(kind, img, lbl, rng):
    """(erased drawing, omitted mask, expected type, in-scope flag, metadata, area mask) or None.
    metadata: type / location / position / size bin of the omission (src/stage2/data/omission_meta.py).
    Large kinds: omitted mask = the erased components (walls, openings, fixtures), expected type "room",
    metadata measured on the whole area (the room + its walls / the rectangle inside the building). area mask =
    that area (large kinds) or the omitted mask (small kinds): what a manual box around the omission covers."""
    if kind in LARGE_KINDS:
        o = (erase_room if kind == "room (whole)" else erase_region)(img, lbl, rng)
        if o is None:
            return None
        out, structure, area = o
        return out, structure, "room", True, omission_meta(lbl, area, kind), area
    if kind in ("door (a)", "door (b)"):
        doors = [d for d in components(lbl == I["door"]) if d.sum() >= 40]
        if not doors:
            return None
        d = doors[rng.integers(len(doors))]
        if kind == "door (a)":
            plan = lbl.copy()
            plan[d] = I["wall"]
            reach = R.reachable_parts(plan)
            cut = any(not (reach & m).any() for _, m in room_instances(plan) if (grown(d, 3) & m).any())
            return draw_wall(img, d, lbl, WALL_PX), d, "door", cut, omission_meta(lbl, d, kind), d
        pair = door_side_pair(lbl, d, 2 * WALL_PX)
        scope = bool(pair and pair[0] != pair[1] and closed_pairs()[pair[0], pair[1]]
                     and {C16.names[p] for p in pair} & {"bedroom", "bathroom"})
        return open_gap(img, d, lbl, WALL_PX), d, "door", scope, omission_meta(lbl, d, kind), d
    if kind == "wall":
        m = wall_piece(lbl, rng)
    elif kind == "room":
        m = room_walls(lbl, rng)
    else:
        cls = C16.group("F") if kind == "fixture" else [I[kind]]
        comps = [c for c in components(np.isin(lbl, cls)) if c.sum() >= 20]
        m = comps[rng.integers(len(comps))] if comps else None
    if m is None or not m.any():
        return None
    typ = C16.names[int(np.bincount(lbl[m]).argmax())] if kind == "fixture" else TYPE[kind]
    out = erase_symbol(img, m, lbl, WALL_PX) if kind in ("window", "fixture") else erase(img, m, lbl, wall_px=WALL_PX)
    return out, m, typ, True, omission_meta(lbl, m, kind), m


def score(props, rm, typ):
    found = correct = False
    for p in props:
        y0, x0, y1, x1 = p.box
        if rm[y0:y1 + 1, x0:x1 + 1].any():
            found = True
            correct |= p.type == typ
    return found, correct


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--completion", required=True)
    ap.add_argument("--drawings", required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: only the first N plans of each group")
    args = ap.parse_args()
    held, _ = build_heldout_cases(args.completion)
    intact = [c["pid"] for c in held if c["mask"] is None]
    omitted = list(dict.fromkeys(c["pid"] for c in held if c["mask"] is not None))
    del held
    if args.limit:
        intact, omitted = intact[:args.limit], omitted[:args.limit]
    z = zipfile.ZipFile(args.drawings)
    rng = np.random.default_rng(5)
    pc = cfg()["parse"]
    methods = {"detector": ("detector",), "detector + C8": ("detector", "C8"),
               "detector + C8 + R5": ("detector", "C8", "R5"),
               "detector + C8 + R5 + ink": ("detector", "C8", "R5", "INK"),
               "+ R6 + R7 (CHECK before the double checks)": ("detector", "C8", "R5", "INK", "R6", "R7"),
               "+ wall ink only (diagnostic; off in the final CHECK)": ("detector", "C8", "R5", "INK", "R6", "R7",
                                                                        "WALLINK"),
               "+ window ink (final CHECK)": ("detector", "C8", "R5", "INK", "R6", "R7", "WINK")}
    choice = yaml.safe_load((PROJECT_ROOT / cfg()["detect"]["settings"]).read_text(encoding="utf-8"))
    choice["extra_rules"] = sorted(set(choice.get("extra_rules", [])) | {"WINK", "WALLINK"})   # every source scored
    rows, t0 = [], time.time()

    def run(pid, img, removed=None, kind=None, typ=None, scope=None, meta=None):  # img: the drawing given to PARSE
        label, _, _ = parse(img, pc)
        parts = detect_parts(label, img, choice)
        row = {"pid": pid, "kind": kind or "intact", "type": typ, "in_scope": scope, "redrawn": ""}
        row.update({f"meta_{k}": v for k, v in (meta or omission_meta(None, None, "intact")).items()})
        rm = None
        if removed is not None:
            rm = cv2.resize(removed.astype(np.uint8), (label.shape[1], label.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            # did the parser draw the removed component back (from context)? share of its pixels
            cls = I["wall"] if typ == "room" else I[typ]
            row["redrawn"] = round(float(np.mean(label[rm] == cls)), 3) if rm.any() else ""
            rm = grown(rm, 3)
        for name, srcs in methods.items():
            props = R.merge([p for s in srcs for p in parts.get(s, [])])
            if rm is None:
                row[name] = len(props)
            else:
                f, c = score(props, rm, typ)
                row[name] = f"{int(f)}{int(c)}"
        rows.append(row)

    for i, pid in enumerate(intact):
        run(pid, read(z, f"{pid}_img.png", cv2.IMREAD_GRAYSCALE))
        print(f"intact {i + 1}/{len(intact)} ({time.time() - t0:.0f} s)", flush=True)
    for j, pid in enumerate(omitted):
        img = read(z, f"{pid}_img.png", cv2.IMREAD_GRAYSCALE)
        lbl = read(z, f"{pid}_lbl.png", cv2.IMREAD_UNCHANGED)
        for k in (2 * j, 2 * j + 1):
            for step in range(len(KINDS)):
                kind = KINDS[(k + step) % len(KINDS)]
                o = omit(kind, img, lbl, rng)
                if o is not None:
                    run(pid, o[0], o[1], kind, o[2], o[3], o[4])
                    break
        print(f"omitted {j + 1}/{len(omitted)} ({time.time() - t0:.0f} s)", flush=True)
    with open(OUT / ("image_eval_smoke.csv" if args.limit else "image_eval_cases.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report(rows, methods, OUT / ("image_eval_smoke.md" if args.limit else "detection_eval.md"))


def report(rows, methods, path):
    ints = [r for r in rows if r["kind"] == "intact"]
    om = [r for r in rows if r["kind"] != "intact"]
    n_k = {k: sum(r["kind"] == k for r in om) for k in KINDS}

    def rec(rs, m):
        if not rs:
            return "-"
        return f"{100 * np.mean([r[m][0] == '1' for r in rs]):.0f}% ({100 * np.mean([r[m][1] == '1' for r in rs]):.0f}%)"
    L = ["", f"## Held-out on DRAWINGS: {TITLE}", "",
         f"{len(ints)} intact drawings + {len(om)} omissions on {len(set(r['pid'] for r in om))} other held-out drawings "
         f"(kinds: " + ", ".join(f"{k} {n_k[k]}" for k in KINDS) + "). Erasure: src/stage2/data/erasure.py; then the "
         "pipeline's PARSE + CHECK (no fill). Found = a proposal overlaps the omission (in brackets: right type).", "",
         "| method | false alarms / intact drawing | mean recall | " + " | ".join(KINDS) + " |",
         "|---|---|---|" + "---|" * len(KINDS)]
    for m in methods:
        fa = np.mean([r[m] for r in ints]) if ints else float("nan")
        per = [np.mean([r[m][0] == "1" for r in om if r["kind"] == k]) for k in KINDS if n_k[k]]
        L.append(f"| {m} | {fa:.2f} | {100 * np.mean(per):.0f}% | " +
                 " | ".join(rec([r for r in om if r["kind"] == k], m) for k in KINDS) + " |")
    fin = list(methods)[-1]
    a_cut = [r for r in om if r["kind"] == "door (a)" and str(r["in_scope"]) == "True"]
    b_scope = [r for r in om if r["kind"] == "door (b)" and str(r["in_scope"]) == "True"]
    L += ["", f"Scope subsets ({fin}): door (a) where the closure cuts a room off: {len(a_cut)} cases, found {rec(a_cut, fin)}; "
          f"door (b) between a bedroom/bathroom and a closed-pair neighbour: {len(b_scope)} cases, found {rec(b_scope, fin)}.",
          "Parser redraws the removed component (mean share of its pixels the parser still labels with the removed "
          "class, room = its walls): " + ", ".join(
              f"{k} {100 * np.mean([float(r['redrawn']) for r in om if r['kind'] == k and r['redrawn'] != '']):.0f}%"
              for k in KINDS if any(r['kind'] == k and r['redrawn'] != '' for r in om)) + ".",
          "Note: label-level numbers above are on ground-truth label maps; these are on parsed drawings, so parser "
          "errors count against detection (and add false alarms)."]
    if om and "meta_location" in om[0]:                  # recall of the final CHECK by omission metadata
        meta_rows = [{**{f: r[f"meta_{f}"] for f in FIELDS}, "r": r} for r in om]
        L += ["", f"Recall of {fin} by omission metadata (found (right type)):"]
        L += breakdowns(meta_rows, {"found": lambda rs: rec([x["r"] for x in rs], fin)})
    with open(OUT / "detection_eval.md", "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
