"""B4 validation with the hole GIVEN (user, 2026-10-21): current models vs the B4 models (256 px large-hole fine-tune)
on val windows, large and small holes. Kaggle CPU kernel.

    python -m src.stage2.complete.b4_val --data <folder with completion_val.zip> [--n 30] -> outputs/stage2/b4_val.md / .csv

Windows: completion_val plans (label level, walls 3 px), seed 77 (training validated on seed 1234):
  large  n windows of 256 x 256 with one large hole (windows.sample_large_window: whole room / region, 2-15% / 5-25%
         of the building; hole = its box + 2-6 px)
  small  n windows of 128 x 128 with one small component hole (windows.sample_window, the original kinds)
Fills (fast sampling as the pipeline, eta 0; the hole's contents are hidden: the map given to the fill has the hole
painted other_room, so nothing of the answer can leak through the restriction):
  large  current models B1 big window, current models B3 tiled (the current large strategy), B4 models B1 big window
  small  current models and B4 models, one 128 window, t0 = fill.t0_fixed (300), no restriction (manual box)
Scores inside the hole vs the ground truth window: tolerant mIoU (hole_ious), validity (thresholded violations <=
the ground truth's, Stage 1 rule), fixed = IoU >= 0.3 of the hole's main class (wall for large / room / region
holes; door / window / railing / wall / the fixture class otherwise) between the fill and the ground truth.
Routing rule (user): small holes stay on the current models unless B4 is at least as good on both tolerant mIoU
and validity; then both are reported and the user decides.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from src.stage2.complete import large_fill, sampler
from src.stage2.complete.data16 import load_plans
from src.stage2.complete.t0_sweep import hole_ious
from src.stage2.complete.windows import sample_large_window, sample_window
from src.stage2.data.classes16 import C16
from src.stage2.pipeline.run import cfg, generator, models
from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "stage2"
I = C16.index


def main_class(kind, gt, hole):
    if kind in ("door", "window", "railing", "wall"):
        return I[kind]
    if kind == "fixture":
        v = gt[hole & np.isin(gt, C16.group("F"))]
        return int(np.bincount(v).argmax()) if len(v) else I["wall"]
    return I["wall"]


def violations(plan, gt, hole):
    lab = torch.from_numpy(plan.astype(np.int64))[None]
    mask = torch.from_numpy(hole.astype(np.float32))[None, None]
    known = torch.nn.functional.one_hot(torch.from_numpy(gt.astype(np.int64))[None], C16.n).permute(0, 3, 1, 2).float()
    return int(sum(v[0] for v in sampler.violation_breakdown(lab, mask, known * (1 - mask)).values()))


def score(pred, gt, hole, kind):
    c = main_class(kind, gt, hole)
    p, g = (pred == c) & hole, (gt == c) & hole
    iou = float((p & g).sum() / max((p | g).sum(), 1))
    return {"tol": round(100 * hole_ious(pred, gt, hole)[1], 1), "valid": violations(pred, gt, hole) <= violations(gt, gt, hole),
            "fix_iou": round(iou, 3), "fixed": iou >= 0.3}


def run(models_used, fn):
    large_fill.OVERRIDE.clear()
    large_fill.OVERRIDE.update(models_used)
    t = time.time()
    out = fn()
    return out, time.time() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--smoke", action="store_true", help="laptop code check: the current models stand in for B4")
    args = ap.parse_args()
    m = dict(models())
    if args.smoke:
        m["b4_unet"], m["b4_diffusion"] = m["unet"], m["diffusion"]
    if "b4_unet" not in m or "b4_diffusion" not in m:
        raise SystemExit("B4 checkpoints not found (configs/stage2/pipeline.yaml checkpoints.b4_*)")
    cur = {"unet": m["unet"], "diffusion": m["diffusion"]}
    b4 = {"unet": m["b4_unet"], "diffusion": m["b4_diffusion"]}
    fc, fast = cfg()["fill"], cfg()["fast"]
    plans = load_plans(Path(args.data) / "completion_val.zip")
    rng = np.random.default_rng(77)
    rows = []
    for size, sampler_fn, win in (("large", sample_large_window, 256), ("small", sample_window, 128)):
        got = 0
        while got < args.n:
            w = sampler_fn(plans[rng.integers(len(plans))][1], rng, win)
            if w is None:
                continue
            gt, hole, kind = w[0], w[1].astype(bool), w[2]
            given = np.where(hole, I["other_room"], gt).astype(np.uint8)
            if size == "large":
                arms = [("current, B1 big window", cur, lambda: large_fill.fill_large(given, hole, "big", fc, fast, generator(0))[0]),
                        ("current, B3 tiled", cur, lambda: large_fill.fill_large(given, hole, "tiled", fc, fast, generator(0))[0]),
                        ("B4, B1 big window", b4, lambda: large_fill.fill_large(given, hole, "big", fc, fast, generator(0))[0])]
            else:
                t0 = fc.get("t0_fixed") or 300
                one = lambda: large_fill.hybrid_window(given, hole, 0, 0, win, t0, fc, fast, generator(0))[0]
                arms = [("current", cur, one), ("B4", b4, one)]
            for name, ms, fn in arms:
                pred, sec = run(ms, fn)
                s = score(np.where(hole, pred, gt), gt, hole, kind)
                rows.append({"size": size, "case": got, "kind": kind, "hole_px": int(hole.sum()), "arm": name,
                             **s, "seconds": round(sec, 1)})
                print(rows[-1], flush=True)
            got += 1
    large_fill.OVERRIDE.clear()
    report(rows, args.n, "_smoke" if args.smoke else "")


def report(rows, n, tag=""):
    L = ["# B4 validation with the hole given (val windows; current models vs B4)", "",
         f"{n} large-hole windows (256 px) + {n} small-hole windows (128 px) from completion_val, seed 77; fast "
         "sampling, eta 0. Scores inside the hole (see src/stage2/complete/b4_val.py).", "",
         "| holes | arm | fixed (IoU >= 0.3) | tol. mIoU | valid | s per hole |", "|---|---|---|---|---|---|"]
    for size in ("large", "small"):
        for arm in dict.fromkeys(r["arm"] for r in rows if r["size"] == size):
            R = [r for r in rows if r["size"] == size and r["arm"] == arm]
            L.append(f"| {size} | {arm} | {sum(r['fixed'] for r in R)} / {len(R)} | {np.mean([r['tol'] for r in R]):.1f} | "
                     f"{100 * np.mean([r['valid'] for r in R]):.0f}% | {np.mean([r['seconds'] for r in R]):.1f} |")
    small = {a: [r for r in rows if r["size"] == "small" and r["arm"] == a] for a in ("current", "B4")}
    ok = all(small[a] for a in small)
    if ok:
        tc, tb = (np.mean([r["tol"] for r in small[a]]) for a in ("current", "B4"))
        vc, vb = (np.mean([r["valid"] for r in small[a]]) for a in ("current", "B4"))
        verdict = ("B4 is at least as good on small holes (tolerant mIoU and validity): ask the user which to use."
                   if tb >= tc and vb >= vc else "B4 is worse on small holes on at least one of tolerant mIoU / validity: "
                   "small holes stay on the current models.")
        L += ["", f"**Small-hole routing:** current {tc:.1f} / {100 * vc:.0f}% valid vs B4 {tb:.1f} / {100 * vb:.0f}% -> "
              + verdict]
    by_kind = sorted({(r["size"], r["kind"]) for r in rows})
    L += ["", "By hole kind (fixed / tol. mIoU per arm):", "", "| holes | kind | n | arms |", "|---|---|---|---|"]
    for size, kind in by_kind:
        R = [r for r in rows if r["size"] == size and r["kind"] == kind]
        arms = dict.fromkeys(r["arm"] for r in R)
        L.append(f"| {size} | {kind} | {len(R) // len(arms)} | " + "; ".join(
            f"{a}: {sum(r['fixed'] for r in R if r['arm'] == a)} / {np.mean([r['tol'] for r in R if r['arm'] == a]):.1f}"
            for a in arms) + " |")
    (OUT / f"b4_val{tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    with open(OUT / f"b4_val{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n".join(L))


if __name__ == "__main__":
    main()
