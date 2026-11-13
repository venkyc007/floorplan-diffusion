"""Novelty step 6 - choosing settings on the TUNING drawings, at map level (no drawings needed).

    python -m src.novelty.eval_maps --set large_tune_0.zip large_tune_1.zip [--n 120] [--t0 500,700,900]
        [--arms b4,sketch,hybrid,auto] [--shard 0/2] [--tag _v1]        -> outputs/novelty/tune_maps{tag}_<shard>.csv
    python -m src.novelty.eval_maps --report outputs/novelty/tune_maps_v1_*.csv --tag _v1
        -> outputs/novelty/tune_maps{tag}.md
    python -m src.novelty.eval_maps --set ... --locator-sweep --tag _v1   -> outputs/novelty/tune_locator{tag}.md

The large-omission set (src/novelty/build_set.py) already holds, for every tuning drawing, the parser's map of the
erased drawing, the ground truth and the omission. So everything that only needs the MAP can be scored here:
  locator sweep   threshold x minimum blob size -> omissions found (a flag's box overlaps the erased structure),
                  found with a box IoU >= 0.5, region flags per intact drawing
  hole given      the hole = the box of the true omission area (what a user would draw). Arms:
                    b4        the shipped large-hole fill (B4 models, big window)            = the baseline
                    sketch    the new U-Net alone
                    hybrid@T  new U-Net sketch -> noised to T -> new diffusion, best of N
  auto (map)      locator flags -> new hybrid fill of every flagged box (no shipped CHECK here: that needs the
                  drawing and is added in the end-to-end evaluation, src/novelty/eval_e2e.py)
Scores (as outputs/stage2/large_fill_compare_v2.md, src/stage2/pipeline/large_fill_compare.score): found, fixed
(wall IoU >= 0.3 at the erased structure), tolerant mIoU and validity inside the omission box; plus violations per
hole, a tolerant wall F1 (2 px) inside the box, and for the hybrid the best sample of the N by wall IoU ("oracle":
how good the fill could be if the verifier picked perfectly - a fairer view when several layouts are valid).
"""

import argparse
import csv
import glob
import time
from pathlib import Path

import cv2
import numpy as np

from src.novelty import pipeline as nov
from src.novelty.data import arrays, load_set
from src.novelty.locate import hit, region_candidates, region_prob
from src.stage2.complete import large_fill
from src.stage2.data.classes16 import C16
from src.stage2.data.erasure import grown
from src.stage2.pipeline import e2e_val, large_fill_compare as lfc
from src.stage2.pipeline import run as base
from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "novelty"
WALL = C16.index["wall"]
REAL = [C16.index[c] for c in base.REAL]


def wall_f1(plan, gt, box, tol=2):
    """Tolerant F1 of wall pixels inside the box: a pixel counts if the other map has wall within `tol` px."""
    sl = (slice(box[0], box[2] + 1), slice(box[1], box[3] + 1))
    p, g = plan[sl] == WALL, gt[sl] == WALL
    if not p.any() or not g.any():
        return 0.0
    prec, rec = (p & grown(g, tol)).sum() / p.sum(), (g & grown(p, tol)).sum() / g.sum()
    return float(2 * prec * rec / max(prec + rec, 1e-9))


def score(plan, parsed, gt, rm, box, holes=None):
    s = lfc.score(plan, parsed, gt, rm, box, holes)
    s.update(viol=lfc.violations(plan, box), viol_gt=lfc.violations(gt, box), wall_f1=round(wall_f1(plan, gt, box), 3))
    return s


def additions(before, after):
    """Number of added real components (wall / door / window / railing / fixture blobs)."""
    return len(e2e_val.components((after != before) & np.isin(after, REAL)))


def case_of(s):
    parsed, gt, om = arrays(s)
    rm = om == 2
    box = lfc.box_of(om > 0)
    hole = np.zeros(parsed.shape, bool)
    hole[box[0]:box[2] + 1, box[1]:box[3] + 1] = True
    return parsed, gt, rm, box, hole


def auto_fill(parsed, m, nc, lc, seed=0):
    """Locator flags -> every flagged box filled in turn with the new hybrid. -> (plan, holes)."""
    holes = nov.region_holes(parsed, lc=lc)
    plan = parsed.copy()
    for h in holes:
        hole = np.zeros(parsed.shape, bool)
        hole[h["box"][0]:h["box"][2] + 1, h["box"][1]:h["box"][3] + 1] = True
        if nov.window_of(hole, nc)[2] > nc["max_window"]:
            continue
        new, _ = nov.hybrid_fill(plan, hole, m["unet"], m["diffusion"], nc["t0"], nc["n_samples"], nc["steps_from_t0"],
                                 base.generator(seed), nc)
        plan = large_fill.region_restrict(plan, new, hole)
    return plan, holes


def run_cases(args):
    samples = load_set(args.set)
    rng = np.random.default_rng(args.seed)
    erased = [i for i in rng.permutation(len(samples)) if samples[i]["version"] > 0]
    intact = [i for i in rng.permutation(len(samples)) if samples[i]["version"] == 0][: args.n_intact]
    si, sn = (int(v) for v in args.shard.split("/"))
    m, nc, lc = nov.models(), nov.cfg()["fill"], nov.cfg()["locate"]
    fc, fast = base.cfg()["fill"], base.cfg()["fast"]
    arms = args.arms.split(",")
    t0s = [int(t) for t in args.t0.split(",")]
    rows, done = [], 0

    def add(s, arm, sc, sec, extra=None):
        rows.append({"pid": s["pid"], "version": s["version"], "kind": s["kind"], "size_bin": s["size_bin"],
                     "size_pct": s["size_pct"], "arm": arm, **{k: sc.get(k, "") for k in
                     ("found", "fixed", "fix_iou", "tol_parse", "tol", "valid_parse", "valid", "viol_gt", "viol", "wall_f1")},
                     "oracle_fix_iou": "", "flags": "", "additions": "", "seconds": round(sec, 1), **(extra or {})})
        print(rows[-1], flush=True)
    for i in erased:
        if done >= args.n:
            break
        s = samples[i]
        parsed, gt, rm, box, hole = case_of(s)
        if nov.window_of(hole, nc)[2] > nc["max_window"]:
            continue
        done += 1
        if (done - 1) % sn != si:
            continue
        if "b4" in arms:
            t = time.time()
            plan, _ = large_fill.fill_large(parsed, hole, "big", fc, fast, base.generator(0))
            add(s, "b4", score(plan, parsed, gt, rm, box), time.time() - t)
        if "sketch" in arms:
            t = time.time()
            plan, _ = nov.hybrid_fill(parsed, hole, m["unet"], None, nc=nc)
            add(s, "sketch", score(large_fill.region_restrict(parsed, plan, hole), parsed, gt, rm, box), time.time() - t)
        if "hybrid" in arms:
            for t0 in t0s:
                t = time.time()
                plan, info = nov.hybrid_fill(parsed, hole, m["unet"], m["diffusion"], t0, args.samples, nc["steps_from_t0"],
                                             base.generator(0), nc, all_samples=True)
                sec = time.time() - t
                rmg = grown(rm, 3)
                oracle = max(e2e_val.right_class_iou(large_fill.region_restrict(parsed, p, hole), WALL, rm, rmg, local=True)
                             for p in info["samples"])
                add(s, f"hybrid@{t0}", score(large_fill.region_restrict(parsed, plan, hole), parsed, gt, rm, box), sec,
                    {"oracle_fix_iou": round(oracle, 3)})
        if "auto" in arms:
            t = time.time()
            plan, holes = auto_fill(parsed, m, nc, lc)
            add(s, "auto", score(plan, parsed, gt, rm, box, holes), time.time() - t,
                {"flags": len(holes), "additions": additions(parsed, plan)})
    if "auto" in arms:
        for j, i in enumerate(intact):
            if j % sn != si:
                continue
            s = samples[i]
            parsed, _, _ = arrays(s)
            t = time.time()
            plan, holes = auto_fill(parsed, m, nc, lc)
            add(s, "auto", {}, time.time() - t, {"flags": len(holes), "additions": additions(parsed, plan)})
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"tune_maps{args.tag}_{si}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, flush=True)


def read_rows(paths):
    rows = []
    for p in sorted(q for pat in paths for q in glob.glob(pat)):
        with open(p, encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    return rows


def truth(v):
    return str(v) == "True"


def report(args):
    rows = read_rows(args.report)
    er = [r for r in rows if r["kind"] != "intact"]
    arms = list(dict.fromkeys(r["arm"] for r in er))
    num = lambda R, k: np.mean([float(r[k]) for r in R if r[k] != ""]) if any(r[k] != "" for r in R) else float("nan")
    pct = lambda R, k: 100 * np.mean([truth(r[k]) for r in R]) if R else float("nan")
    n = len({(r["pid"], r["version"]) for r in er})
    L = [f"# Large holes on the TUNING drawings, map level{args.tag}", "",
         f"{n} erased versions of tuning val drawings (parser's map of the erased drawing; kinds: " +
         ", ".join(f"{k} {len({(r['pid'], r['version']) for r in er if r['kind'] == k})}" for k in dict.fromkeys(r["kind"] for r in er)) +
         "). Hole given = the box of the true omission area; auto = the locator's flags, each filled by the new "
         "hybrid. Fast sampling. See src/novelty/eval_maps.py for the scores.", "",
         "| arm | fixed | wall IoU | oracle wall IoU (best of N) | tol. mIoU (parse -> fill) | wall F1 (2 px) | valid | violations (truth) | found | s per hole |",
         "|---|---|---|---|---|---|---|---|---|---|"]

    def line(name, R):
        found = f"{pct(R, 'found'):.0f}%" if any(r["found"] != "" for r in R) else "given"
        return (f"| {name} | {pct(R, 'fixed'):.0f}% | {num(R, 'fix_iou'):.3f} | "
                f"{num(R, 'oracle_fix_iou'):.3f} | {num(R, 'tol_parse'):.1f} -> {num(R, 'tol'):.1f} | {num(R, 'wall_f1'):.3f} | "
                f"{pct(R, 'valid'):.0f}% | {num(R, 'viol'):.1f} ({num(R, 'viol_gt'):.1f}) | {found} | {num(R, 'seconds'):.1f} |")
    for a in arms:
        L.append(line(a, [r for r in er if r["arm"] == a]))
    for field in ("kind", "size_bin"):
        L += ["", f"By {field.replace('_', ' ')} (fixed / tol. mIoU / valid):", "",
              f"| {field} | n | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
        for v in dict.fromkeys(r[field] for r in er):
            sub = [r for r in er if r[field] == v]
            L.append(f"| {v} | {len(sub) // max(len(arms), 1)} | " + " | ".join(
                f"{pct([r for r in sub if r['arm'] == a], 'fixed'):.0f}% / {num([r for r in sub if r['arm'] == a], 'tol'):.1f} / "
                f"{pct([r for r in sub if r['arm'] == a], 'valid'):.0f}%" for a in arms) + " |")
    it = [r for r in rows if r["kind"] == "intact"]
    if it:
        L += ["", f"Intact drawings ({len(it)}; auto): region flags per drawing {num(it, 'flags'):.2f}, drawings with a flag "
              f"{100 * np.mean([float(r['flags']) > 0 for r in it]):.0f}%, added components per drawing {num(it, 'additions'):.2f}."]
    path = OUT / f"tune_maps{args.tag}.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def locator_sweep(args):
    samples = load_set(args.set)
    locs = nov.models()["locator"]
    sets = {f"locator {i}": [m] for i, m in enumerate(locs)}
    if len(locs) > 1:
        sets["average of all"] = locs
    L = [f"# Region locator on the TUNING drawings{args.tag}", "",
         f"{sum(s['version'] > 0 for s in samples)} erased versions + {sum(s['version'] == 0 for s in samples)} intact "
         "drawings (parser's maps). found = a flag's box overlaps the omission area; box IoU >= 0.5 = that flag's box "
         "also matches the area's box; flags / intact = region flags per intact drawing (false alarms); stray = flags "
         "on an erased drawing that miss the omission.", ""]
    for name, ms in sets.items():
        probs = [region_prob(ms, arrays(s)[0], base.DEVICE) for s in samples]
        L += [f"## {name}", "", "| threshold | min blob px | found | found, medium | found, large | box IoU >= 0.5 | flags / intact drawing | intact drawings flagged | stray flags / erased drawing |",
              "|---|---|---|---|---|---|---|---|---|"]
        for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            for ma in (100, 150, 300):
                f, f50, fa, fl, stray, by = [], [], [], [], [], {"medium": [], "large": []}
                for s, p in zip(samples, probs):
                    flags = region_candidates(p, th, ma)
                    area = arrays(s)[2] > 0
                    if s["version"] == 0:
                        fa.append(len(flags))
                        fl.append(len(flags) > 0)
                    else:
                        f.append(hit(flags, area))
                        f50.append(hit(flags, area, 0.5))
                        by.get(s["size_bin"], []).append(f[-1])
                        stray.append(sum(1 for c in flags if not area[c.box[0]:c.box[2] + 1, c.box[1]:c.box[3] + 1].any()))
                L.append(f"| {th} | {ma} | {100 * np.mean(f):.0f}% | {100 * np.mean(by['medium'] or [0]):.0f}% | "
                         f"{100 * np.mean(by['large'] or [0]):.0f}% | {100 * np.mean(f50):.0f}% | {np.mean(fa):.2f} | "
                         f"{100 * np.mean(fl):.0f}% | {np.mean(stray):.2f} |")
        L.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"tune_locator{args.tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def gate_sweep(args):
    """The "would the fill change anything?" gate (pipeline.sketch_change) on the tuning drawings: for the locator
    flags at a few thresholds, the new wall pixels the U-Net sketch draws in the flagged box; then found / false
    alarms when only flags with at least G new wall pixels are kept."""
    samples = load_set(args.set)
    m, nc = nov.models(), nov.cfg()["fill"]
    probs = [region_prob(m["locator"], arrays(s)[0], base.DEVICE) for s in samples]
    L = [f"# Region flags kept only if the fill would change the area{args.tag}", "",
         "Locator = the configured one (average if several), blob >= 150 px. change = wall pixels the large-hole U-Net "
         "draws in the flagged box that are not within 2 px of a wall of the parsed map. found = an omission with a "
         "kept flag on it; flags / intact = kept flags per intact drawing.", "",
         "| threshold | min change (wall px) | found | found, medium | found, large | flags / intact drawing | intact drawings flagged | stray flags / erased drawing |",
         "|---|---|---|---|---|---|---|---|"]
    rows = []
    for th in (0.3, 0.5, 0.7):
        recs = []                                          # per sample: [(hit, change)]
        for s, p in zip(samples, probs):
            parsed, _, om = arrays(s)
            area = om > 0
            recs.append([(bool(area[c.box[0]:c.box[2] + 1, c.box[1]:c.box[3] + 1].any()),
                          nov.sketch_change(parsed, c.box, m["unet"], nc)) for c in region_candidates(p, th, 150)])
            for hit_, ch in recs[-1]:
                rows.append({"threshold": th, "pid": s["pid"], "version": s["version"], "size_bin": s["size_bin"],
                             "on_omission": hit_, "change": ch})
        for g in (0, 10, 20, 40, 80, 160):
            f, fa, stray, by = [], [], [], {"medium": [], "large": []}
            for s, r in zip(samples, recs):
                kept = [h for h, ch in r if ch >= g]
                if s["version"] == 0:
                    fa.append(len(kept))
                else:
                    f.append(any(kept))
                    by.get(s["size_bin"], []).append(f[-1])
                    stray.append(sum(1 for h in kept if not h))
            L.append(f"| {th} | {g} | {100 * np.mean(f):.0f}% | {100 * np.mean(by['medium'] or [0]):.0f}% | "
                     f"{100 * np.mean(by['large'] or [0]):.0f}% | {np.mean(fa):.2f} | {100 * np.mean([v > 0 for v in fa]):.0f}% | "
                     f"{np.mean(stray):.2f} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"tune_gate{args.tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    with open(OUT / f"tune_gate{args.tag}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", nargs="+")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--n-intact", type=int, default=100)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--arms", default="b4,sketch,hybrid,auto")
    ap.add_argument("--t0", default="500,700,900")
    ap.add_argument("--samples", type=int, default=4, help="N of the hybrid arm (best of N; the oracle looks at all N)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--locator-sweep", action="store_true")
    ap.add_argument("--gate-sweep", action="store_true")
    ap.add_argument("--report", nargs="+")
    args = ap.parse_args()
    if args.report:
        return report(args)
    if args.locator_sweep:
        return locator_sweep(args)
    if args.gate_sweep:
        return gate_sweep(args)
    run_cases(args)


if __name__ == "__main__":
    main()
