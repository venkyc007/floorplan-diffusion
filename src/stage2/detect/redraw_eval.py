"""Does the parser draw an erased component back? (user, 2026-10-19; erasure v2 = whole symbol removed)

    python -m src.stage2.detect.redraw_eval --zip parser_val.zip --heldout heldout_ids.txt [--n 100]
        -> outputs/stage2/redraw_rate.md, outputs/stage2/redraw_cases.csv

The first n typed HELD-OUT val drawings (file order). In each, separately: one door erased as kind (a) (wall drawn
across), one door as kind (b) (gap left), one window, one fixture (random component >= 40 / 20 px, seed 7),
with src/stage2/data/erasure.py. Each erased drawing goes through the pipeline's PARSE (parser v1, rescaling as in
the app). Redraw rate = share of the removed component's pixels (resized to the parse) that the parser still labels
with the removed class (fixture: its own fixture class). For comparison, the same share on the INTACT drawing
(how much of the component the parser finds when it is there).
"""

import argparse
import csv
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

from src.stage2.constraints.rules import components
from src.stage2.data.classes16 import C16
from src.stage2.data.erasure import draw_wall, erase_symbol, open_gap
from src.stage2.pipeline.run import cfg, parse
from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "stage2"
I = C16.index
KINDS = ["door (a)", "door (b)", "window", "fixture"]
WALL_PX = 8


def read(z, name, flag):
    return cv2.imdecode(np.frombuffer(z.read(name), np.uint8), flag)


def pick(kind, lbl, rng):
    if kind.startswith("door"):
        comps = [c for c in components(lbl == I["door"]) if c.sum() >= 40]
    elif kind == "window":
        comps = [c for c in components(lbl == I["window"]) if c.sum() >= 40]
    else:
        comps = [c for c in components(np.isin(lbl, C16.group("F"))) if c.sum() >= 20]
    return comps[rng.integers(len(comps))] if comps else None


def erased(kind, img, m, lbl):
    if kind == "door (a)":
        return draw_wall(img, m, lbl, WALL_PX)
    if kind == "door (b)":
        return open_gap(img, m, lbl, WALL_PX)
    return erase_symbol(img, m, lbl, WALL_PX)


def share(label, m, cls):
    rm = cv2.resize(m.astype(np.uint8), (label.shape[1], label.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return (float(np.mean(label[rm] == cls)), int(rm.sum())) if rm.any() else (None, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    keep = set(Path(args.heldout).read_text(encoding="utf-8").split())
    z = zipfile.ZipFile(args.zip)
    names = sorted(n[:-8] for n in z.namelist() if n.endswith("_img.png") and n[:-8] in keep)
    rng = np.random.default_rng(7)
    pc = cfg()["parse"]
    rows, done, t0 = [], 0, time.time()
    for n in names:
        if done >= args.n:
            break
        lbl = read(z, f"{n}_lbl.png", cv2.IMREAD_UNCHANGED)
        if (lbl == 255).any():
            continue
        img = read(z, f"{n}_img.png", cv2.IMREAD_GRAYSCALE)
        intact, _, _ = parse(img, pc)
        for kind in KINDS:
            m = pick(kind, lbl, rng)
            if m is None:
                continue
            cls = int(np.bincount(lbl[m]).argmax())
            label, _, _ = parse(erased(kind, img, m, lbl), pc)
            r, px = share(label, m, cls)
            r0, _ = share(intact, m, cls)
            if r is not None:
                rows.append({"pid": n, "kind": kind, "class": C16.names[cls], "px": px,
                             "redrawn": round(r, 4), "found_intact": round(r0, 4)})
        done += 1
        print(f"{done}/{args.n} {n} ({time.time() - t0:.0f} s)", flush=True)
    with open(OUT / "redraw_cases.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    L = ["# Redraw rate: does the parser (v1) label an ERASED component back? (erasure v2, held-out val)", "",
         f"{done} held-out drawings; per type one component erased at a time (whole symbol: door zone / box + 40%). "
         "Redrawn = share of the removed component's pixels the parser still labels with its class (mean over cases; "
         "pooled over pixels in brackets). Found when intact = the same share on the original drawing.", "",
         "| type | cases | redrawn | cases > 50% redrawn | found when intact |", "|---|---|---|---|---|"]
    groups = KINDS + ["door (a + b)"]
    for k in groups:
        rs = [r for r in rows if (r["kind"] == k or (k == "door (a + b)" and r["kind"].startswith("door")))]
        if not rs:
            continue
        red = np.array([r["redrawn"] for r in rs])
        px = np.array([r["px"] for r in rs])
        found = np.array([r["found_intact"] for r in rs])
        L.append(f"| {k} | {len(rs)} | {100 * red.mean():.0f}% ({100 * (red * px).sum() / px.sum():.0f}%) | "
                 f"{100 * np.mean(red > 0.5):.0f}% | {100 * found.mean():.0f}% |")
    fx = [r for r in rows if r["kind"] == "fixture"]
    if fx:
        L += ["", "Fixtures by class: " + ", ".join(
            f"{c} {100 * np.mean([r['redrawn'] for r in fx if r['class'] == c]):.0f}% ({sum(r['class'] == c for r in fx)})"
            for c in sorted({r['class'] for r in fx})) + "."]
    L += ["", "Before (overnight, erasure v1, image_eval on other omissions): door (a) 55%, door (b) 85%, window 77%, "
          "fixture 8%."]
    (OUT / "redraw_rate.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
