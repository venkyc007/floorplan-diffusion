"""How much does the scale mismatch cost the FILL? (audit 2026-10-21; Kaggle CPU kernel stage2_scale_diag)

    python -m src.stage2.complete.scale_diag --data <folder with completion_val.zip> [--n 30]
        -> outputs/stage2/scale_diag.md / .csv

Background (docs/decisions.md, 2026-10-21 night): the fill models were trained on completion-set maps (scale 0.375 of
the parser set, walls ~4 px), but run_pipeline makes its maps at f = 3 / wall_thickness = 3 / 9.55 = 0.314, i.e.
R = 8 / 9.55 = 0.838 of the training scale.

Test: the SAME val plan and the SAME hole, filled
  (a) at the TRAINING scale (the completion-set map as it is), and
  (b) at the PIPELINE scale: the whole plan and the hole box resized by R (nearest neighbour), as run_pipeline does.
Ground-truth labels, so parse errors play no part: only the scale changes. The window is cut around the hole at
each scale, so (b) sees more plan context in the same window, as in the pipeline.
Holes: n small (windows.deletion kinds, box + 2-6 px; 128 px window) and n large (whole room / region, box + 2-6 px;
256 px window) on completion_val plans, seed 91, the window centred on the hole.
Fills (fast settings, as validated; generator seed 0 for every fill):
  small  "U-Net sketch" (K2a alone) and "final" (K2a sketch -> t0 300 -> K1 constrained diffusion)
  large  "final" (B4 models, one big window, region restriction: large_fill.fill_large "big")
Scores inside the hole (src/stage2/complete/b4_val.score):
  A. own scale  each fill against the ground truth at its own scale: the model's quality at each scale, without any
                resizing loss (validity uses the constants of the training scale, as the pipeline's verifier does)
  B. training   both at the TRAINING scale: the pipeline-scale result is pasted into the resized plan, resized back by
                1/R and compared with the original ground truth (the same target for (a) and (b)); this includes
                the loss from resizing itself, shown by the ceiling: the ground truth resized by R and back, no model
Differences (b) - (a) with paired bootstrap 95% CIs.
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from src.stage2.complete import large_fill
from src.stage2.complete.b4_val import score
from src.stage2.complete.data16 import load_plans
from src.stage2.complete.windows import LARGE_TYPES, TYPES, _bbox, deletion, large_deletion
from src.stage2.data.classes16 import C16
from src.stage2.final_test import boot_ci
from src.stage2.pipeline.run import cfg, generator, models
from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "stage2"
I = C16.index
R = 8 / 9.55                       # pipeline map scale / completion-set (training) scale
WIN = {"small": 128, "large": 256}


def plan_hole(label, rng, size):
    """(hole box (y0, x0, y1, x1) in plan px, kind) as windows.sample_window / sample_large_window define it, or None."""
    for kind in rng.permutation(LARGE_TYPES if size == "large" else TYPES):
        removed = large_deletion(label, kind, rng) if size == "large" else deletion(label, kind, rng, 128)
        if removed is None or not removed.any():
            continue
        y0, y1, x0, x1 = _bbox(removed)
        if size == "large" or kind != "region":                  # Stage 1 rule: box + 2-6 px per side
            pad = rng.integers(2, 7, size=4)
            y0, y1, x0, x1 = y0 - pad[0], y1 + pad[1], x0 - pad[2], x1 + pad[3]
        h, w = label.shape
        y0, x0, y1, x1 = max(0, y0), max(0, x0), min(h - 1, y1), min(w - 1, x1)
        limit = WIN[size] - (16 if size == "large" else 0)
        if y1 - y0 + 1 > limit or x1 - x0 + 1 > limit:
            continue
        return (int(y0), int(x0), int(y1), int(x1)), str(kind)
    return None


def crop(label, box, win):
    """win x win window centred on the box (background outside the plan) -> (window, hole mask, origin)."""
    y0, x0, y1, x1 = box
    oy, ox = (y0 + y1) // 2 - win // 2, (x0 + x1) // 2 - win // 2
    out = np.full((win, win), I["background"], np.uint8)
    h, w = label.shape
    sy0, sx0, sy1, sx1 = max(0, oy), max(0, ox), min(h, oy + win), min(w, ox + win)
    out[sy0 - oy:sy1 - oy, sx0 - ox:sx1 - ox] = label[sy0:sy1, sx0:sx1]
    hole = np.zeros((win, win), bool)
    hole[max(0, y0 - oy):y1 - oy + 1, max(0, x0 - ox):x1 - ox + 1] = True
    return out, hole, (oy, ox)


def paste(label, win, origin):
    """The plan with the window written back at its origin (parts outside the plan dropped)."""
    out = label.copy()
    oy, ox = origin
    h, w = label.shape
    sy0, sx0, sy1, sx1 = max(0, oy), max(0, ox), min(h, oy + win.shape[0]), min(w, ox + win.shape[1])
    out[sy0:sy1, sx0:sx1] = win[sy0 - oy:sy1 - oy, sx0 - ox:sx1 - ox]
    return out


def resize(label, shape):
    return cv2.resize(label, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def scale_box(box, f, shape):
    y0, x0, y1, x1 = box
    h, w = shape
    return (int(np.floor(y0 * f)), int(np.floor(x0 * f)), min(h - 1, int(np.ceil(y1 * f))), min(w - 1, int(np.ceil(x1 * f))))


def fills(given, hole, size, fc, fast, cur):
    """{arm: (prediction window, seconds)} for one window."""
    out = {}
    if size == "small":
        large_fill.OVERRIDE.clear()
        large_fill.OVERRIDE.update(cur)
        t = time.time()
        pred, _, sketch = large_fill.hybrid_window(given, hole, 0, 0, given.shape[0], fc.get("t0_fixed") or 300, fc, fast,
                                                   generator(0))
        sec = time.time() - t
        out["U-Net sketch"], out["final"] = (sketch, 0.0), (pred, sec)
    else:
        large_fill.OVERRIDE.clear()                       # -> B4 models (fill.large_models: b4)
        t = time.time()
        out["final"] = (large_fill.fill_large(given, hole, "big", fc, fast, generator(0))[0], time.time() - t)
    large_fill.OVERRIDE.clear()
    return out


def one_case(plan, box, kind, size, fc, fast, cur):
    """Rows for one hole: both scales, every arm, main / own-scale scores; plus the resampling ceiling."""
    win = WIN[size]
    small_shape = (max(8, round(plan.shape[0] * R)), max(8, round(plan.shape[1] * R)))
    plan_r = resize(plan, small_shape)
    box_r = scale_box(box, R, small_shape)
    gt, hole, origin = crop(plan, box, win)
    gt_r, hole_r, origin_r = crop(plan_r, box_r, win)
    rows = []
    for scale, g, hl in (("training", gt, hole), ("pipeline", gt_r, hole_r)):
        given = np.where(hl, I["other_room"], g).astype(np.uint8)
        for arm, (pred, sec) in fills(given, hl, size, fc, fast, cur).items():
            pred = np.where(hl, pred, g).astype(np.uint8)
            own = score(pred, g, hl, kind)
            if scale == "training":
                main = own
            else:                                          # back to the training scale, same target as (a)
                back = resize(paste(plan_r, pred, origin_r), plan.shape)
                pb = crop(back, box, win)[0]
                main = score(np.where(hole, pb, gt), gt, hole, kind)
            rows.append({"size": size, "kind": kind, "arm": arm, "scale": scale, "hole_px": int(hole.sum()),
                         "tol": main["tol"], "fixed": main["fixed"], "valid_own": own["valid"], "tol_own": own["tol"],
                         "fixed_own": own["fixed"], "seconds": round(sec, 1)})
    rt = crop(resize(plan_r, plan.shape), box, win)[0]
    c = score(np.where(hole, rt, gt), gt, hole, kind)
    rows.append({"size": size, "kind": kind, "arm": "resampling ceiling", "scale": "pipeline", "hole_px": int(hole.sum()),
                 "tol": c["tol"], "fixed": c["fixed"], "valid_own": "", "tol_own": "", "fixed_own": "", "seconds": 0.0})
    return rows


def report(rows, n):
    pc = lambda v: f"{100 * np.mean(v):.0f}%"
    L = ["# Scale mismatch: what it costs the FILL (hole given, val)", "",
         f"{n} small + {n} large holes on completion_val plans (seed 91). The SAME plan and hole filled at the training "
         f"scale and at the pipeline's scale (plan resized by R = {R:.3f}, nearest neighbour, as run_pipeline's maps "
         "are). Ground-truth labels: only the scale changes. Fast settings. Script: src/stage2/complete/scale_diag.py.",
         "", "How to read it: A compares each fill with the truth at its own scale. Its tolerant mIoU leans towards the "
         "pipeline scale (the 1 px tolerance is relatively larger when walls are thinner), so 'fixed' (main-class "
         "component IoU >= 0.3) and validity are the fairer columns there. B scores both at the training scale; it "
         "includes the loss from resizing itself (see the ceiling row), which leans towards the training scale. A real "
         "scale fix would avoid that resizing loss (FILL would get its own map at the training scale), so A's fixed / "
         "valid columns are the best guide to what a fix could gain.", "",
         "## A. Each fill against the truth at its own scale (the model's quality at each scale)", "",
         "| holes | arm | n | tol. mIoU training scale | tol. mIoU pipeline scale | difference [95% CI] | fixed training / pipeline | valid training / pipeline |",
         "|---|---|---|---|---|---|---|---|"]
    arms = lambda size: dict.fromkeys(r["arm"] for r in rows if r["size"] == size and r["arm"] != "resampling ceiling")
    pair = lambda size, arm: ([r for r in rows if (r["size"], r["arm"], r["scale"]) == (size, arm, "training")],
                              [r for r in rows if (r["size"], r["arm"], r["scale"]) == (size, arm, "pipeline")])

    def diff(a, b, key):
        d = np.array([float(y[key]) for y in b]) - np.array([float(x[key]) for x in a])
        lo, hi = boot_ci(d)
        return f"{d.mean():+.1f} [{lo:+.1f}, {hi:+.1f}]"

    for size in ("small", "large"):
        for arm in arms(size):
            a, b = pair(size, arm)
            L.append(f"| {size} | {arm} | {len(a)} | {np.mean([x['tol_own'] for x in a]):.1f} | "
                     f"{np.mean([y['tol_own'] for y in b]):.1f} | {diff(a, b, 'tol_own')} | "
                     f"{pc([x['fixed_own'] for x in a])} / {pc([y['fixed_own'] for y in b])} | "
                     f"{pc([x['valid_own'] for x in a])} / {pc([y['valid_own'] for y in b])} |")
    L += ["", "## B. Both scored at the training scale (the pipeline-scale fill resized back to it)", "",
          "Includes the loss from resizing itself; the ceiling row is the truth resized by R and back with no model.", "",
          "| holes | arm | n | tol. mIoU training scale | tol. mIoU pipeline scale | difference [95% CI] | fixed training / pipeline |",
          "|---|---|---|---|---|---|---|"]
    for size in ("small", "large"):
        for arm in arms(size):
            a, b = pair(size, arm)
            L.append(f"| {size} | {arm} | {len(a)} | {np.mean([x['tol'] for x in a]):.1f} | {np.mean([y['tol'] for y in b]):.1f} | "
                     f"{diff(a, b, 'tol')} | {pc([x['fixed'] for x in a])} / {pc([y['fixed'] for y in b])} |")
        c = [r for r in rows if r["size"] == size and r["arm"] == "resampling ceiling"]
        if c:
            L.append(f"| {size} | resampling ceiling (truth resized and back) | {len(c)} | 100.0 | {np.mean([r['tol'] for r in c]):.1f} | "
                     f"{np.mean([r['tol'] for r in c]) - 100:+.1f} | 100% / {pc([r['fixed'] for r in c])} |")
    L += ["", "By hole kind (final arm, own-scale tol. mIoU training -> pipeline scale):", "",
          "| holes | kind | n | training | pipeline |", "|---|---|---|---|---|"]
    for size, kind in sorted({(r["size"], r["kind"]) for r in rows}):
        a = [r["tol_own"] for r in rows if (r["size"], r["kind"], r["arm"], r["scale"]) == (size, kind, "final", "training")]
        b = [r["tol_own"] for r in rows if (r["size"], r["kind"], r["arm"], r["scale"]) == (size, kind, "final", "pipeline")]
        if a:
            L.append(f"| {size} | {kind} | {len(a)} | {np.mean(a):.1f} | {np.mean(b):.1f} |")
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    m = dict(models())
    if "b4_unet" not in m or "b4_diffusion" not in m:
        raise SystemExit("B4 checkpoints not found (configs/stage2/pipeline.yaml checkpoints.b4_*)")
    cur = {"unet": m["unet"], "diffusion": m["diffusion"]}
    fc, fast = cfg()["fill"], cfg()["fast"]
    plans = load_plans(Path(args.data) / "completion_val.zip")
    rng = np.random.default_rng(91)
    rows = []
    for size in ("small", "large"):
        got = 0
        while got < args.n:
            plan = plans[rng.integers(len(plans))][1]
            h = plan_hole(plan, rng, size)
            if h is None:
                continue
            rows += one_case(plan, h[0], h[1], size, fc, fast, cur)
            got += 1
            print(size, got, h[1], [(r["arm"], r["scale"], r["tol"]) for r in rows[-4:]], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"scale_diag{args.tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    L = report(rows, args.n)
    (OUT / f"scale_diag{args.tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
