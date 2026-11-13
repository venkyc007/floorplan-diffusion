"""Novelty step 7 - end to end on DRAWINGS: the shipped pipeline vs the hybrid pipeline, all omission sizes.

    python -m src.novelty.eval_e2e --zip parser_val.zip --split heldout [--limit N] [--shard 0/2] [--tag _v1]
        -> outputs/novelty/e2e_<split>{tag}_<shard>.csv
    python -m src.novelty.eval_e2e --report "outputs/novelty/e2e_heldout_v1_*.csv" --tag _heldout_v1
        -> outputs/novelty/e2e{tag}.md

Drawings: the HELD-OUT val drawings (configs/stage2/val_split/heldout_ids.txt, never used to choose a setting)
minus the 12 check-set drawings and untyped plans, in a fixed random order (seed 30). --split tuning runs the same
on the TUNING drawings (used to choose settings; those numbers are optimistic and labelled so). --split test
(parser_test.zip, all typed test drawings) is prepared for ONE run of the frozen hybrid and refuses to start without
--test-approved-by-user.
Per drawing: the intact drawing, and up to three separate cases - one small, one medium, one large omission
(src/stage2/detect/omission_mix.omit_in_bin, the generator of the Stage 2 test; seed 31 + drawing index), erased on
the drawing itself. Nothing about the omission is given to the automatic arms.
Arms
  shipped auto      run_pipeline(fast=True): the frozen Stage 2 pipeline                      (the baseline)
  hybrid auto       src/novelty/pipeline.run: + region locator + new large-hole fill          (the novelty)
  locator + B4      the locator's region flags filled by the SHIPPED large-hole models (ablation: which part helps)
  shipped manual    medium / large only: the box of the true omission area given, shipped B4 fill
  hybrid manual     the same box, the new large-hole fill
Scores: src/stage2/pipeline/e2e_val.score, the function behind the Stage 2 test table (found, fixed = found and
wall IoU >= 0.3 at the erased structure, false additions, tolerant mIoU and validity in the omission region), plus
"region flag" (a region-type flag overlaps the omission). Intact drawings: flags, region flags and added components.
Intervals: Wilson 95%; differences: paired bootstrap over drawings (2,000 resamples).
"""

import argparse
import csv
import glob
import time
import zipfile

import cv2
import numpy as np

from src.novelty import pipeline as nov
from src.stage2.complete import large_fill
from src.stage2.constraints.rules import components
from src.stage2.data.classes16 import C16
from src.stage2.data.erasure import grown
from src.stage2.detect.omission_mix import omit_in_bin
from src.stage2.pipeline import e2e_val, large_fill_compare as lfc
from src.stage2.pipeline import run as base
from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "novelty"
SPLIT = PROJECT_ROOT / "configs" / "stage2" / "val_split"
REAL = [C16.index[c] for c in base.REAL]
BINS = ("small", "medium", "large")
AUTO = {"shipped auto": {}, "hybrid auto": {"region_fn": nov.region_holes, "large_fill_fn": nov.fill_region},
        "locator + B4": {"region_fn": nov.region_holes}}
FIELDS = ["pid", "target", "kind", "size_bin", "size_pct", "arm", "found", "region_flag", "fixed", "fix_iou",
          "false_adds", "tol_before", "tol_after", "valid_before", "valid_after", "viol_gt", "viol_after",
          "n_flags", "n_region_flags", "added", "seconds"]


def drawing_ids(split, z):
    if split == "test":                                   # the official test drawings (parser_test.zip): every plan
        ids = sorted(n[:-8] for n in z.namelist() if n.endswith("_img.png"))
    else:
        ids = (SPLIT / f"{split}_ids.txt").read_text(encoding="utf-8").split()
    order = [ids[i] for i in np.random.default_rng(30).permutation(len(ids))]
    return [p for p in order if p not in e2e_val.CHECK_SET and f"{p}_img.png" in z.namelist()]


def added_components(res):
    marked = res.get("marked", np.zeros(res["parsed"].shape, bool))
    return len(components(((res["completed"] != res["parsed"]) | marked) & np.isin(res["completed"], REAL)))


def manual_result(parsed, plan, box):
    """A run_pipeline-like result for a fill with the hole given (so e2e_val.score can score it)."""
    return {"parsed": parsed, "completed": plan, "holes": [{"type": "manual", "rules": [], "box": box}],
            "marked": np.zeros(parsed.shape, bool)}


def row_of(pid, target, o, arm, res, sec):
    if o is None:                                         # intact drawing
        return {"pid": pid, "target": "intact", "kind": "intact", "arm": arm, "n_flags": len(res["holes"]),
                "n_region_flags": sum(h["type"] == "region" for h in res["holes"]), "added": added_components(res),
                "seconds": round(sec, 1)}
    s = e2e_val.score(res, o[1], o["lbl"], o[2])
    rmg = grown(s["rm"], 3)
    over = [h for h in res["holes"] if rmg[h["box"][0]:h["box"][2] + 1, h["box"][1]:h["box"][3] + 1].any()]
    return {"pid": pid, "target": target, "kind": o[4]["type"], "size_bin": o[4]["size_bin"], "size_pct": o[4]["size_pct"],
            "arm": arm, "found": s["found"], "region_flag": any(h["type"] == "region" for h in over), "fixed": s["fixed"],
            "fix_iou": s["fix_iou"], "false_adds": s["false_adds"], "tol_before": s["tol_before"], "tol_after": s["tol_after"],
            "valid_before": s["valid_before"], "valid_after": s["valid_after"], "viol_gt": s["viol_gt"],
            "viol_after": s["viol_after"], "n_flags": len(res["holes"]),
            "n_region_flags": sum(h["type"] == "region" for h in res["holes"]), "added": added_components(res),
            "seconds": round(sec, 1)}


def run_cases(args):
    z = zipfile.ZipFile(args.zip)
    read = lambda n, f: cv2.imdecode(np.frombuffer(z.read(n), np.uint8), f)
    ids = drawing_ids(args.split, z)
    si, sn = (int(v) for v in args.shard.split("/"))
    arms = [a.strip() for a in args.arms.split(",")]
    fc, fast = base.cfg()["fill"], base.cfg()["fast"]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"e2e_{args.split}{args.tag}_{si}.csv"
    n_done, shown = 0, []
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        w.writeheader()

        def emit(row):
            w.writerow(row)
            f.flush()
            print(row, flush=True)
        for idx, pid in enumerate(ids):
            if idx % sn != si:
                continue
            lbl = read(f"{pid}_lbl.png", cv2.IMREAD_UNCHANGED)
            if (lbl == 255).any():
                continue
            if args.limit and n_done >= args.limit:
                break
            n_done += 1
            img = read(f"{pid}_img.png", cv2.IMREAD_GRAYSCALE)
            rng = np.random.default_rng([31, idx])
            todo = [("intact", None)] + [(b, omit_in_bin(b, img, lbl, rng)) for b in BINS if b in args.bins]
            for target, o in todo:
                if target != "intact" and o is None:
                    continue
                image = img if o is None else o[0]
                oc = None if o is None else {0: o[0], 1: o[1], 2: o[2], 4: o[4], 5: o[5], "lbl": lbl}
                parsed, keep = None, None
                if o is not None and target != "small" and len(shown) < args.figure:
                    keep = {"pid": pid, "kind": o[4]["type"], "size": f"{o[4]['size_bin']} ({o[4]['size_pct']}%)",
                            "image": image, "lbl": lbl, "area": o[5], "plans": {}, "fixed": {}}
                    shown.append(keep)
                for arm in arms:
                    if arm not in AUTO:
                        continue
                    t = time.time()
                    res = base.run_pipeline(image, fast=True, **AUTO[arm])
                    emit(row_of(pid, target, oc, arm, res, time.time() - t))
                    parsed = res["parsed"]
                    if keep is not None:
                        keep["parsed"], keep["plans"][arm] = parsed, res["completed"]
                        keep["fixed"][arm] = e2e_val.score(res, o[1], lbl, o[2])["fixed"]
                if o is None or target == "small" or parsed is None:
                    continue
                area = lfc.to_parse_scale(o[5], parsed.shape) | lfc.to_parse_scale(o[1], parsed.shape)
                if not area.any():
                    continue
                box = lfc.box_of(area)
                hole = np.zeros(parsed.shape, bool)
                hole[box[0]:box[2] + 1, box[1]:box[3] + 1] = True
                if "shipped manual" in arms:
                    t = time.time()
                    plan, _ = large_fill.fill_large(parsed, hole, "big", fc, fast, base.generator(0))
                    emit(row_of(pid, target, oc, "shipped manual", manual_result(parsed, plan, box), time.time() - t))
                if "hybrid manual" in arms:
                    t = time.time()
                    plan, _ = nov.fill_region(parsed, hole, base.generator(0), fast)
                    r = row_of(pid, target, oc, "hybrid manual", manual_result(parsed, plan, box), time.time() - t)
                    emit(r)
                    if keep is not None:
                        keep["plans"]["hybrid manual"], keep["fixed"]["hybrid manual"] = plan, r["fixed"]
    print("wrote", path, flush=True)
    if shown:
        figure(shown, OUT / f"e2e_{args.split}{args.tag}_examples_{si}.png")


def figure(cases, path):
    """The first cases of the run, as they come (no picking): erased drawing | parser's map | each arm's completed
    map | ground truth, cropped around the omission (red box = the true omission area, for the reader only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = list(cases[0]["plans"])
    fig, axes = plt.subplots(len(cases), len(arms) + 3, figsize=(4.2 * (len(arms) + 3), 4.2 * len(cases)), squeeze=False)
    for c, ax in zip(cases, axes):
        H, W = c["parsed"].shape
        area = lfc.to_parse_scale(c["area"], (H, W))
        y0, x0, y1, x1 = lfc.box_of(area)
        m = max(24, (y1 - y0) // 2, (x1 - x0) // 2)
        sl = (slice(max(0, y0 - m), min(H, y1 + m + 1)), slice(max(0, x0 - m), min(W, x1 + m + 1)))
        fy, fx = c["image"].shape[0] / H, c["image"].shape[1] / W
        isl = (slice(int(sl[0].start * fy), int(sl[0].stop * fy)), slice(int(sl[1].start * fx), int(sl[1].stop * fx)))
        gt = cv2.resize(c["lbl"], (W, H), interpolation=cv2.INTER_NEAREST)
        ax[0].imshow(c["image"][isl], cmap="gray", vmin=0, vmax=255)
        ax[0].set_title(f"{c['pid']}\n{c['kind']}, {c['size']}: drawing given", fontsize=9)
        panels = [("parser's map", c["parsed"])] + [(f"{a}: {'fixed' if c['fixed'].get(a) else 'not fixed'}", c["plans"][a])
                                                    for a in arms] + [("ground truth", gt)]
        for a, (title, plan) in zip(ax[1:], panels):
            a.imshow(C16.colourise(plan[sl]), interpolation="nearest")
            a.set_title(title, fontsize=9)
            a.add_patch(plt.Rectangle((x0 - sl[1].start - .5, y0 - sl[0].start - .5), x1 - x0 + 1, y1 - y0 + 1,
                                      fill=False, ec="red", lw=1))
        for a in ax:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=60)
    print("wrote", path, flush=True)


# ---------------------------------------------------------------- report

def wilson(k, n, zc=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + zc ** 2 / n
    c, h = (p + zc ** 2 / (2 * n)) / d, zc * np.sqrt(p * (1 - p) / n + zc ** 2 / (4 * n ** 2)) / d
    return 100 * (c - h), 100 * (c + h)


def paired_boot(a, b, n=2000, seed=0):
    """Mean difference b - a over paired cases, with a 95% bootstrap interval."""
    d = np.asarray(b, float) - np.asarray(a, float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def report(args):
    rows = []
    for p in sorted(q for pat in args.report for q in glob.glob(pat)):
        with open(p, encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    T = lambda v: str(v) == "True"
    om = [r for r in rows if r["kind"] != "intact"]
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    mean = lambda R, k: np.mean([float(r[k]) for r in R]) if R else float("nan")
    L = [f"# End to end on drawings: shipped pipeline vs hybrid pipeline{args.tag}", "",
         f"{len({r['pid'] for r in rows})} val drawings ({args.split_note}). One omission per case, erased on the drawing; "
         "the automatic arms are given nothing. fixed = found and wall IoU >= 0.3 at the erased structure "
         "(src/stage2/pipeline/e2e_val.score, as in the Stage 2 test). Brackets: Wilson 95% intervals. "
         "Script: src/novelty/eval_e2e.py.", ""]
    for b in BINS + ("medium + large",):
        sub = [r for r in om if (r["size_bin"] in ("medium", "large") if b == "medium + large" else r["size_bin"] == b)]
        if not sub:
            continue
        L += [f"## {b} omissions", "",
              "| arm | n | found | region flag | fixed | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for a in arms:
            R = [r for r in sub if r["arm"] == a]
            if not R:
                continue
            k = sum(T(r["fixed"]) for r in R)
            lo, hi = wilson(k, len(R))
            L.append(f"| {a} | {len(R)} | {100 * np.mean([T(r['found']) for r in R]):.0f}% | "
                     f"{100 * np.mean([T(r['region_flag']) for r in R]):.0f}% | **{100 * k / len(R):.0f}%** [{lo:.0f}, {hi:.0f}] | "
                     f"{mean(R, 'tol_before'):.1f} -> {mean(R, 'tol_after'):.1f} | "
                     f"{100 * np.mean([T(r['valid_before']) for r in R]):.0f}% -> {100 * np.mean([T(r['valid_after']) for r in R]):.0f}% | "
                     f"{mean(R, 'viol_after'):.1f} ({mean(R, 'viol_gt'):.1f}) | {mean(R, 'false_adds'):.2f} | {mean(R, 'seconds'):.1f} |")
        key = lambda r: (r["pid"], r["target"])
        for a, c in (("shipped auto", "hybrid auto"), ("shipped manual", "hybrid manual"), ("locator + B4", "hybrid auto")):
            A = {key(r): r for r in sub if r["arm"] == a}
            C = {key(r): r for r in sub if r["arm"] == c}
            ks = sorted(set(A) & set(C))
            if not ks:
                continue
            parts = []
            for field, scale, name in (("fixed", 100, "fixed, points"), ("tol_after", 1, "tol. mIoU"), ("valid_after", 100, "valid, points")):
                val = (lambda r: float(T(r[field]))) if field != "tol_after" else (lambda r: float(r[field]))
                d, lo, hi = paired_boot([val(A[k]) for k in ks], [val(C[k]) for k in ks])
                parts.append(f"{name} {scale * d:+.1f} [{scale * lo:+.1f}, {scale * hi:+.1f}]")
            L.append("")
            L.append(f"{c} minus {a} (paired, {len(ks)} cases): " + "; ".join(parts) + ".")
        L.append("")
    kinds = [k for k in dict.fromkeys(r["kind"] for r in om)]
    L += ["## By omission kind (fixed)", "", "| kind | n | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
    for kd in kinds:
        sub = [r for r in om if r["kind"] == kd]
        n = len({(r["pid"], r["target"]) for r in sub})
        cell = lambda a: (f"{100 * np.mean([T(r['fixed']) for r in sub if r['arm'] == a]):.0f}%"
                          if any(r["arm"] == a for r in sub) else "-")
        L.append(f"| {kd} | {n} | " + " | ".join(cell(a) for a in arms) + " |")
    it = [r for r in rows if r["kind"] == "intact"]
    if it:
        L += ["", "## Intact drawings (false alarms)", "",
              "| arm | n | flags / drawing | region flags / drawing | drawings with a region flag | added components / drawing |",
              "|---|---|---|---|---|---|"]
        for a in arms:
            R = [r for r in it if r["arm"] == a]
            if R:
                L.append(f"| {a} | {len(R)} | {mean(R, 'n_flags'):.2f} | {mean(R, 'n_region_flags'):.2f} | "
                         f"{100 * np.mean([float(r['n_region_flags']) > 0 for r in R]):.0f}% | {mean(R, 'added'):.2f} |")
    (OUT / f"e2e{args.tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip")
    ap.add_argument("--split", default="heldout", choices=("heldout", "tuning", "test"))
    ap.add_argument("--test-approved-by-user", action="store_true",
                    help="required with --split test: the test set is used once, only after the user says so")
    ap.add_argument("--limit", type=int, default=None, help="only the first N drawings of this shard")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--arms", default="shipped auto,hybrid auto,locator + B4,shipped manual,hybrid manual")
    ap.add_argument("--bins", default="small,medium,large")
    ap.add_argument("--tag", default="")
    ap.add_argument("--figure", type=int, default=0, help="also save a figure of the first K medium / large cases")
    ap.add_argument("--report", nargs="+")
    ap.add_argument("--split-note", default="held-out: never used to choose a setting")
    args = ap.parse_args()
    if args.report:
        return report(args)
    if args.split == "test" and not args.test_approved_by_user:
        raise SystemExit("The Stage 2 test set is closed: run --split test only after the user approved this one run "
                         "(docs/decisions.md, novelty section) and pass --test-approved-by-user.")
    run_cases(args)


if __name__ == "__main__":
    main()
