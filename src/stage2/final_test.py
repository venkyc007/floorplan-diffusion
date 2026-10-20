"""Final Stage 2 TEST evaluation (run ONCE, on the CubiCasa test split; user 2026-10-21). Built for the Kaggle
notebook kaggle/notebooks/stage2_final_test (two T4s: one process per GPU, cases split by index; resumable).

    python -m src.stage2.final_test --parser-zip parser_test.zip --completion-zip completion_test.zip --out DIR \
        --part 0 --parts 2          # one shard (env STAGE2_DEVICE=cuda, CUDA_VISIBLE_DEVICES=0)
    python -m src.stage2.final_test --report --out DIR                 # merge the shards -> DIR/final_test.md
    add --smoke for a 2-case-per-section code check

Settings: configs/stage2/test_eval.yaml (counts, seed, size mix). Every case's result is appended to
DIR/part<k>.jsonl as soon as it is done; a re-run skips the cases already there (resume after a time-out: attach the
previous version's output, the notebook copies its jsonl files back first).

Sections
  fill      the FILL alone with the hole GIVEN, on completion_test windows, size-balanced (a third each of small /
            medium / large holes, measured as % of the plan's building): small = the original component holes
            (windows.sample_window, 128 px), medium / large = whole rooms / regions (windows.sample_large_window,
            256 px). ABLATION: the same U-Net sketch (k2a) finished by the CONSTRAINED diffusion (k1_constrained) vs
            the CONTROL diffusion (k1_control, identical training without the constraint losses); same seed per
            case. Evaluation sampling (fill.n_samples, DDIM 50, t0 = t0_fixed for small holes, the size-adaptive
            t0 for large ones). Scores inside the hole: tolerant mIoU, validity (violations <= the ground truth's),
            violations per hole, fixed (IoU >= 0.3 of the hole's main class). Paired bootstrap 95% CI of the
            differences.
  detect    PARSE + CHECK (configs/stage2/detector_threshold.yaml) on test drawings: n intact (false alarms per
            drawing) + n with one size-balanced image-level omission (src/stage2/detect/omission_mix.py):
            found / right type, by type, location, position and size bin.
  e2e       the whole pipeline (run_pipeline, evaluation settings) on further test drawings, one size-balanced
            omission each: found, right type, fixed, false additions, tolerant mIoU and validity in the omission
            region, seconds per drawing; broken down by the omission metadata.
"""

import argparse
import glob
import json
import os
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.utils import PROJECT_ROOT

CFG = yaml.safe_load((PROJECT_ROOT / "configs" / "stage2" / "test_eval.yaml").read_text(encoding="utf-8"))
BINS = ("small", "medium", "large")


def read(z, name, flag):
    return cv2.imdecode(np.frombuffer(z.read(name), np.uint8), flag)


# ---------------------------------------------------------------- case lists (identical in every shard)

def fill_cases(completion_zip, n_per_bin, seed):
    from src.stage2.complete.data16 import load_plans
    from src.stage2.complete.windows import sample_large_window, sample_window
    from src.stage2.data.classes16 import C16
    plans = load_plans(completion_zip)
    rng = np.random.default_rng(seed)
    out, counts, tries = [], {b: 0 for b in BINS}, 0
    while min(counts.values()) < n_per_bin and tries < 200 * n_per_bin:
        tries += 1
        pid, lab = plans[rng.integers(len(plans))]
        building = max(int((~np.isin(lab, C16.group("OUT"))).sum()), 1)
        want = min(counts, key=counts.get)
        w = sample_window(lab, rng, 128) if want == "small" else sample_large_window(lab, rng, 256)
        if w is None:
            continue
        pct = 100.0 * w[1].sum() / building
        b = "small" if pct < 2.7 else ("medium" if pct < 7.7 else "large")
        if counts[b] >= n_per_bin:
            continue
        counts[b] += 1
        out.append({"pid": pid, "gt": w[0], "hole": w[1].astype(bool), "kind": w[2], "bin": b, "pct": round(pct, 2)})
    return out


def drawing_ids(parser_zip):
    z = zipfile.ZipFile(parser_zip)
    names = sorted(n[:-8] for n in z.namelist() if n.endswith("_img.png"))
    return [n for n in names if not (read(z, f"{n}_lbl.png", cv2.IMREAD_UNCHANGED) == 255).any()]


# ---------------------------------------------------------------- per-case work

def do_fill(c, models_, fc):
    from src.stage2.complete import large_fill
    from src.stage2.complete.b4_val import score, violations
    from src.stage2.data.classes16 import C16
    from src.stage2.pipeline.run import generator
    I = C16.index
    gt, hole = c["gt"], c["hole"]
    given = np.where(hole, I["other_room"], gt).astype(np.uint8)
    side = gt.shape[0]
    t0 = (fc.get("t0_fixed") or 300) if c["bin"] == "small" else large_fill.size_t0(given, hole, fc)
    res = {}
    for arm in ("constrained", "control"):
        large_fill.OVERRIDE.clear()
        large_fill.OVERRIDE.update({"unet": models_["unet"], "diffusion": models_[arm]})
        t = time.time()
        pred = large_fill.hybrid_window(given, hole, 0, 0, side, t0, fc, None, generator(c["idx"]))[0]
        pred = np.where(hole, pred, gt)
        res[arm] = {**score(pred, gt, hole, c["kind"]), "violations": violations(pred, gt, hole),
                    "seconds": round(time.time() - t, 1)}
    large_fill.OVERRIDE.clear()
    return {"pid": c["pid"], "kind": c["kind"], "bin": c["bin"], "pct": c["pct"], "t0": int(t0),
            "gt_violations": violations(gt, gt, hole), **res}


def do_detect(z, pid, target, rng):
    from src.stage2.data.erasure import grown
    from src.stage2.data.omission_meta import omission_meta
    from src.stage2.detect import rules as R
    from src.stage2.detect.omission_mix import omit_balanced
    from src.stage2.pipeline.run import cfg, detect_holes, parse
    img = read(z, f"{pid}_img.png", cv2.IMREAD_GRAYSCALE)
    lbl = read(z, f"{pid}_lbl.png", cv2.IMREAD_UNCHANGED)
    meta, removed, typ, used = omission_meta(None, None, "intact"), None, None, ""
    if target:
        o = omit_balanced(target, img, lbl, rng)
        if o is None:
            return None
        used, (img, removed, typ, _, meta, _) = o
    label, _, _ = parse(img, cfg()["parse"])
    holes = detect_holes(label, img)
    row = {"pid": pid, "target_bin": target or "", "bin_used": used, "type": typ, **{f"meta_{k}": v for k, v in meta.items()},
           "flags": "; ".join(f"{h.type} ({'+'.join(h.rules)})" for h in holes)}
    if removed is None:
        row["false_alarms"] = len(holes)
    else:
        rm = grown(cv2.resize(removed.astype(np.uint8), label.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0, 3)
        over = [h for h in holes if rm[h.box[0]:h.box[2] + 1, h.box[1]:h.box[3] + 1].any()]
        row.update(found=bool(over), right_type=any(h.type == typ or (typ == "room" and h.type == "region") for h in over))
    return row


def do_e2e(z, pid, target, rng):
    from src.stage2.detect.omission_mix import omit_balanced
    from src.stage2.pipeline import e2e_val
    from src.stage2.pipeline.run import run_pipeline
    img = read(z, f"{pid}_img.png", cv2.IMREAD_GRAYSCALE)
    lbl = read(z, f"{pid}_lbl.png", cv2.IMREAD_UNCHANGED)
    o = omit_balanced(target, img, lbl, rng)
    if o is None:
        return None
    used, (img, removed, typ, _, meta, _) = o
    t = time.time()
    res = run_pipeline(img, fast=False)
    sec = time.time() - t
    s = e2e_val.score(res, removed, lbl, typ)
    return {"pid": pid, "target_bin": target, "bin_used": used, "type": typ, **{f"meta_{k}": v for k, v in meta.items()},
            **{k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in s.items() if k not in ("gt", "rm")},
            "messages": res["messages"], "seconds": round(sec, 1)}


# ---------------------------------------------------------------- shard runner

def done_keys(out):
    keys = set()
    for f in glob.glob(str(Path(out) / "part*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            keys.add((r["section"], r["idx"]))
    return keys


def run_shard(args):
    from src.stage2.pipeline.run import build16, cfg, models
    import torch
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = done_keys(out)
    log = open(out / f"part{args.part}.jsonl", "a", encoding="utf-8")
    smoke = 2 if args.smoke else None
    fc = cfg()["fill"]

    def emit(section, idx, row):
        log.write(json.dumps({"section": section, "idx": idx, **row}, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
        log.flush()

    mine = lambda i: i % args.parts == args.part
    # fill ablation
    m = dict(models())
    ck = torch.load(PROJECT_ROOT / cfg()["checkpoints"]["diffusion_control"], map_location="cpu", weights_only=False)
    ctrl = build16(ck["config"]["model"])
    ctrl.load_state_dict(ck["model"])
    from src.stage2.pipeline.run import DEVICE
    fm = {"unet": m["unet"], "constrained": m["diffusion"], "control": ctrl.eval().to(DEVICE)}
    cases = fill_cases(args.completion_zip, smoke or CFG["n_fill_per_bin"], CFG["seed"])
    for i, c in enumerate(cases):
        if mine(i) and ("fill", i) not in done:
            emit("fill", i, do_fill({**c, "idx": i}, fm, fc))
            print(f"[part {args.part}] fill {i + 1}/{len(cases)}", flush=True)
    # detection + end to end on test drawings (disjoint drawing sets)
    z = zipfile.ZipFile(args.parser_zip)
    ids = drawing_ids(args.parser_zip)
    order = [ids[i] for i in np.random.default_rng(CFG["seed"]).permutation(len(ids))]
    n_int, n_om, n_e2e = (smoke, smoke, smoke) if smoke else (CFG["n_detect_intact"], CFG["n_detect_omitted"], CFG["n_e2e"])
    from src.stage2.detect.omission_mix import plan_bins
    plan = ([("detect", None)] * n_int + [("detect", b) for b in plan_bins(n_om)] + [("e2e", b) for b in plan_bins(n_e2e)])
    for i, (section, target) in enumerate(plan):
        if i >= len(order) or not mine(i) or (section, i) in done:
            continue
        rng = np.random.default_rng(CFG["seed"] + i)          # per case: the same omission in any shard / re-run
        row = (do_detect if section == "detect" else do_e2e)(z, order[i], target, rng)
        emit(section, i, row if row is not None else {"pid": order[i], "skipped": True})
        print(f"[part {args.part}] {section} {i + 1}/{len(plan)}", flush=True)
    log.close()


# ---------------------------------------------------------------- report

def boot_ci(d, n=2000, seed=0):
    d = np.asarray(d, float)
    if len(d) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(out):
    from src.stage2.data.omission_meta import FIELDS, breakdowns
    rows = [json.loads(l) for f in sorted(glob.glob(str(Path(out) / "part*.jsonl"))) for l in open(f, encoding="utf-8")]
    rows = list({(r["section"], r["idx"]): r for r in rows}.values())
    fill = [r for r in rows if r["section"] == "fill"]
    det = [r for r in rows if r["section"] == "detect" and not r.get("skipped")]
    e2e = [r for r in rows if r["section"] == "e2e" and not r.get("skipped")]
    pc = lambda v: f"{100 * np.mean(v):.0f}%" if len(v) else "-"
    L = ["# Stage 2 final TEST evaluation (CubiCasa test split, run once)", "",
         f"Settings: configs/stage2/test_eval.yaml; cases: fill {len(fill)}, detection {len(det)}, end to end {len(e2e)} "
         f"(skipped: {sum(1 for r in rows if r.get('skipped'))}).", "",
         "## 1. FILL with the hole given: constrained vs control diffusion (ablation)", "",
         "Same U-Net sketch and seed per case; inside the hole. CI = paired bootstrap 95% of (constrained - control).", "",
         "| size bin | n | tol. mIoU constrained / control | diff [95% CI] | validity constrained / control | "
         "violations per hole constrained / control | diff [95% CI] | fixed constrained / control |",
         "|---|---|---|---|---|---|---|---|"]
    for b in BINS + ("all",):
        R = [r for r in fill if b == "all" or r["bin"] == b]
        if not R:
            continue
        g = lambda arm, k: [float(r[arm][k]) for r in R]
        dt = np.array(g("constrained", "tol")) - np.array(g("control", "tol"))
        dv = np.array(g("constrained", "violations")) - np.array(g("control", "violations"))
        lo, hi = boot_ci(dt)
        lo2, hi2 = boot_ci(dv)
        L.append(f"| {b} | {len(R)} | {np.mean(g('constrained', 'tol')):.1f} / {np.mean(g('control', 'tol')):.1f} | "
                 f"{dt.mean():+.1f} [{lo:+.1f}, {hi:+.1f}] | {pc(g('constrained', 'valid'))} / {pc(g('control', 'valid'))} | "
                 f"{np.mean(g('constrained', 'violations')):.2f} / {np.mean(g('control', 'violations')):.2f} | "
                 f"{dv.mean():+.2f} [{lo2:+.2f}, {hi2:+.2f}] | {pc(g('constrained', 'fixed'))} / {pc(g('control', 'fixed'))} |")
    ints = [r for r in det if r["target_bin"] == ""]
    oms = [r for r in det if r["target_bin"] != ""]
    L += ["", "## 2. CHECK (detection) on test drawings", "",
          f"False alarms per intact drawing: {np.mean([r['false_alarms'] for r in ints]):.2f} ({len(ints)} drawings). "
          f"Omissions: {len(oms)}, found {pc([r['found'] for r in oms])} (right type {pc([r['right_type'] for r in oms])})."]
    mrows = [{**{f: r[f"meta_{f}"] for f in FIELDS}, "r": r} for r in oms]
    L += breakdowns(mrows, {"found": lambda rs: pc([x["r"]["found"] for x in rs]),
                            "right type": lambda rs: pc([x["r"]["right_type"] for x in rs])})
    L += ["", "## 3. End to end on test drawings", ""]
    if e2e:
        t = lambda k: pc([bool(r[k]) for r in e2e])
        L.append(f"{len(e2e)} drawings: found {t('found')}, right type {t('right_type')}, fixed {t('fixed')} (by addition "
                 f"{t('fixed_add')}), false additions {np.mean([r['false_adds'] for r in e2e]):.2f} per drawing, tolerant "
                 f"mIoU {np.mean([r['tol_before'] for r in e2e]):.1f} -> {np.mean([r['tol_after'] for r in e2e]):.1f}, validity "
                 f"{t('valid_before')} -> {t('valid_after')}, {np.mean([r['seconds'] for r in e2e]):.0f} s per drawing (GPU).")
        erows = [{**{f: r[f"meta_{f}"] for f in FIELDS}, "r": r} for r in e2e]
        L += breakdowns(erows, {"found": lambda rs: pc([x["r"]["found"] for x in rs]),
                                "fixed": lambda rs: pc([x["r"]["fixed"] for x in rs]),
                                "false adds": lambda rs: f"{np.mean([x['r']['false_adds'] for x in rs]):.2f}",
                                "tol. mIoU after": lambda rs: f"{np.mean([x['r']['tol_after'] for x in rs]):.1f}"})
    bins_used = {b: sum(r["bin_used"] == b for r in oms + e2e) for b in BINS}
    L += ["", f"Size mix realised (detection omissions + end to end): {bins_used}."]
    (Path(out) / "final_test.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parser-zip")
    ap.add_argument("--completion-zip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report(args.out)
    else:
        run_shard(args)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
