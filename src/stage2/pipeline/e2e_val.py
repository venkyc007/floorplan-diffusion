"""Quick end-to-end validation on 12 NEW held-out val drawings (user, 2026-10-20).

    python -m src.stage2.pipeline.e2e_val --zip parser_val.zip [--limit N] [--tag _v2]
        -> outputs/stage2/e2e_val{tag}.png (input | parse + flags | result | ground truth), .md, .csv
    python -m src.stage2.pipeline.e2e_val --compare old.csv new.csv --tag _v2
        -> appends an old vs new table (same drawings) to e2e_val{tag}.md (laptop, after the kernel)

Drawings: the held-out val plans (configs/stage2/val_split/heldout_ids.txt; never used to choose a setting),
minus the 12 check-set drawings and untyped plans, in a fixed random order (seed 20; no picking by hand).
For each kind in turn, the first 2 unused drawings that have such a component: door (a), door (b), window,
wall, fixture, room. One omission per drawing, made on the DRAWING with erasure v2 (src/stage2/detect/
image_eval.omit, the same generator as the held-out tables), with its metadata (type, location, position,
size bin; src/stage2/data/omission_meta.py).
Pipeline: run_pipeline(fast=True) = parse (v2B) -> final CHECK (configs/stage2/detector_threshold.yaml) ->
fill (hybrid fast mode, t0 = 300, N = 2, eta from configs/stage2/pipeline.yaml, sketch prior, type restriction).

Scores (completion scale; omission = the removed pixels resized to the parsed map):
  found        a flag (hole box) overlaps the omission (grown 3 px); right type = such a flag has its type
  fixed        (user, 2026-10-20) found AND the FINAL plan has the right class at the omission with IoU >= 0.3,
               whoever drew it (fill, or the parser: "symbol missing in the drawing"). door / window / fixture:
               the final plan's components of that class touching the omission (grown 3 px); wall / room
               (class wall): the final plan's wall pixels within 3 px of the omission (walls are one network)
  fixed by addition   the earlier definition: only the fill's added pixels of the right class count
  false adds   added or marked ("symbol missing") components (wall / door / window / railing / fixture) that do
               not touch the omission
  tolerant mIoU, validity   inside the omission region (its box + 4 px), final plan vs the ground truth
               (resized to the parsed map); validity = no more thresholded violations than the ground truth
               (Stage 1 rule; sampler.violation_breakdown on a 128 window). Also shown for the parse before
               the fill, so the fill's own effect is visible.
  time         seconds for the whole pipeline on the drawing (Kaggle CPU)
  diagnosis    for drawings where the fill added violations: the violations per rule of the parse, the RAW hybrid
               output (before the type restriction) and the final plan, so each added rule is attributed to the
               hybrid itself or to the restriction
"""

import argparse
import csv
import time
import zipfile

import cv2
import numpy as np
import torch

from src.stage2.complete import sampler
from src.stage2.complete.t0_sweep import hole_ious
from src.stage2.constraints.rules import components
from src.stage2.data.classes16 import C16
from src.stage2.data.erasure import grown
from src.stage2.data.omission_meta import FIELDS, breakdowns
from src.stage2.detect.image_eval import omit
from src.stage2.pipeline.run import REAL, cfg, run_pipeline, window_around
from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "stage2"
I = C16.index
PLAN = ["door (a)", "door (b)", "window", "wall", "fixture", "room"]
PER_KIND = 2
HELDOUT = PROJECT_ROOT / "configs" / "stage2" / "val_split" / "heldout_ids.txt"
CHECK_SET = {"colorful-11261", "colorful-1148", "colorful-1261", "colorful-13689", "colorful-13513", "colorful-13804",
             "colorful-13820", "colorful-13824", "colorful-13826", "colorful-14036", "colorful-14126", "colorful-14147"}
FIX_IOU = 0.3
PAD = 4


def read(z, name, flag):
    return cv2.imdecode(np.frombuffer(z.read(name), np.uint8), flag)


def region_box(m, shape, pad=PAD):
    ys, xs = np.nonzero(m)
    return (max(0, ys.min() - pad), max(0, xs.min() - pad), min(shape[0] - 1, ys.max() + pad),
            min(shape[1] - 1, xs.max() + pad))


def rule_violations(plan, box, win=128):
    """{rule: thresholded violations} of `plan` inside `box` (a 128 window around it); zero rules left out."""
    w, _, hole = window_around(plan, box, win)
    lab = torch.from_numpy(w.astype(np.int64))[None]
    mask = torch.from_numpy(hole).float()[None, None]
    known = torch.nn.functional.one_hot(lab, C16.n).permute(0, 3, 1, 2).float() * (1 - mask)
    return {k: int(v[0]) for k, v in sampler.violation_breakdown(lab, mask, known).items() if v[0]}


def violations(plan, box, win=128):
    """Thresholded violations of `plan` inside `box` (a 128 window around it)."""
    return sum(rule_violations(plan, box, win).values())


def raw_plan(res):
    """The plan with every hole's RAW hybrid output (before the type restriction) pasted back in order."""
    plan = res["parsed"].copy()
    for h in res["holes"]:
        if "raw" in h:
            oy, ox = h["raw"]["origin"]
            hh, ww = h["raw"]["hybrid"].shape
            plan[oy:oy + hh, ox:ox + ww] = h["raw"]["hybrid"]
    return plan


def diagnose(res, box, before, after):
    """Rules with more violations after the fill than in the parse; each attributed to the hybrid (raw
    output has them too) or to the type restriction (raw output does not)."""
    raw = rule_violations(raw_plan(res), box)
    out = []
    for k in sorted(set(after) | set(before)):
        if after.get(k, 0) > before.get(k, 0):
            cause = "restriction" if raw.get(k, 0) <= before.get(k, 0) else "hybrid"
            out.append(f"{k} {before.get(k, 0)}->{after[k]} (raw {raw.get(k, 0)}: {cause})")
    return "; ".join(out)


def right_class_iou(plan, cls, rm, rmg, local):
    """IoU of the omitted component with the plan's pixels of class cls at it: whole components touching the
    omission, or (local, walls) only the pixels within the grown omission."""
    near = ((plan == cls) & rmg) if local else _whole(plan == cls, rmg)
    return float((near & rm).sum() / max((near | rm).sum(), 1))


def _whole(mask, seed):
    """The components of mask that contain any seed pixel."""
    out = np.zeros(mask.shape, bool)
    for m in components(mask):
        if (m & seed).any():
            out |= m
    return out


def score(res, removed, lbl, typ):
    comp, parsed = res["completed"], res["parsed"]
    H, W = comp.shape
    rm = cv2.resize(removed.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    gt = cv2.resize(lbl, (W, H), interpolation=cv2.INTER_NEAREST)
    if not rm.any():                                   # vanished when resized: use the nearest pixel block
        ys, xs = np.nonzero(removed)
        rm[min(H - 1, int(ys.mean() * H / lbl.shape[0])), min(W - 1, int(xs.mean() * W / lbl.shape[1]))] = True
    rmg = grown(rm, 3)
    over = [h for h in res["holes"] if rmg[h["box"][0]:h["box"][2] + 1, h["box"][1]:h["box"][3] + 1].any()]
    cls = I["wall"] if typ == "room" else I[typ]
    marked = res.get("marked", np.zeros(rm.shape, bool))
    added = ((comp != parsed) | marked) & np.isin(comp, [I[c] for c in REAL])
    near = np.zeros(rm.shape, bool)
    for m in components((comp != parsed) & (comp == cls)):      # earlier definition: the fill's additions only
        if (m & rmg).any():
            near |= m
    iou_add = float((near & rm).sum() / max((near | rm).sum(), 1))
    iou = right_class_iou(comp, cls, rm, rmg, local=cls == I["wall"])
    false = sum(1 for m in components(added) if not (m & rmg).any())
    box = region_box(rm, rm.shape)
    reg = np.zeros(rm.shape, bool)
    reg[box[0]:box[2] + 1, box[1]:box[3] + 1] = True
    r_gt, r_after, r_before = rule_violations(gt, box), rule_violations(comp, box), rule_violations(parsed, box)
    v_gt, v_after, v_before = sum(r_gt.values()), sum(r_after.values()), sum(r_before.values())
    return {"found": bool(over), "right_type": any(h["type"] == typ for h in over), "fix_iou": round(iou, 3),
            "fixed": bool(over) and iou >= FIX_IOU, "fix_iou_add": round(iou_add, 3), "fixed_add": iou_add >= FIX_IOU,
            "marked_on_spot": bool((marked & rmg).any()), "false_adds": false,
            "diagnosis": diagnose(res, box, r_before, r_after) if v_after > v_before else "",
            "gap": "; ".join(f"{h['gap_len']} px" for h in over if "gap_len" in h),
            "tol_before": round(100 * hole_ious(parsed, gt, reg)[1], 1),
            "tol_after": round(100 * hole_ious(comp, gt, reg)[1], 1),
            "valid_before": v_before <= v_gt, "valid_after": v_after <= v_gt,
            "viol_gt": v_gt, "viol_before": v_before, "viol_after": v_after,
            "flags": "; ".join(f"{h['type']} ({'+'.join(h['rules'])})" for h in res["holes"]) or "none",
            "flags_on_spot": len(over), "gt": gt, "rm": rm}


def pick(z, args):
    held = HELDOUT.read_text(encoding="utf-8").split()
    order = [held[i] for i in np.random.default_rng(20).permutation(len(held))]
    order = [p for p in order if p not in CHECK_SET]
    rng = np.random.default_rng(20)
    used, cases = set(), []
    for kind in PLAN:
        got = 0
        for pid in order:
            if got >= PER_KIND:
                break
            if pid in used:
                continue
            lbl = read(z, f"{pid}_lbl.png", cv2.IMREAD_UNCHANGED)
            if (lbl == 255).any():                     # untyped plan
                continue
            img = read(z, f"{pid}_img.png", cv2.IMREAD_GRAYSCALE)
            o = omit(kind, img, lbl, rng)
            if o is None:
                continue
            used.add(pid)
            got += 1
            cases.append({"pid": pid, "kind": kind, "img": o[0], "removed": o[1], "typ": o[2], "meta": o[4],
                          "lbl": lbl})
            if args.limit and len(cases) >= args.limit:
                return cases
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: only the first N drawings")
    ap.add_argument("--tag", default="")
    ap.add_argument("--compare", nargs=2, metavar=("OLD_CSV", "NEW_CSV"))
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare, args.tag)
    z = zipfile.ZipFile(args.zip)
    cases = pick(z, args)
    for c in cases:
        t = time.time()
        c["res"] = run_pipeline(c["img"], fast=True)
        c["seconds"] = round(time.time() - t, 1)
        c["score"] = score(c["res"], c["removed"], c["lbl"], c["typ"])
        s = c["score"]
        print(c["pid"], c["kind"], f"{c['seconds']} s", {k: v for k, v in s.items() if k not in ("gt", "rm")},
              c["res"]["messages"], flush=True)
    report(cases, args.tag)
    figure(cases, args.tag)


def report(cases, tag):
    yn = lambda b: "yes" if b else "no"
    fc = cfg()["fill"]
    n = len(cases)
    S = [c["score"] for c in cases]
    L = [f"# End-to-end validation on {n} new held-out val drawings{tag}", "",
         f"Pipeline: parse v2B -> final CHECK -> fill (fast hybrid, t0 = {fc.get('t0_fixed')}, N = "
         f"{cfg()['fast']['n_samples']}, eta = {fc['eta_guidance']:g}, sketch prior, type restriction); Kaggle CPU. "
         "One omission per drawing (erasure v2). Drawings: held-out val plans minus the check set, fixed random "
         "order (seed 20). Scores at the parsed map's scale; tolerant mIoU / validity inside the omission region "
         "(box + 4 px), before -> after the fill, vs the ground truth.", "",
         "fixed = a flag there AND the final plan has the right class at the omission (IoU >= 0.3), whoever drew "
         "it; fixed by addition = the earlier definition (the fill's own additions only). gap = the measured wall "
         "break at a door flag (src/stage2/detect/flag_gap.py).", "",
         "| # | drawing | omission | location | position | size | flags (type + source) | found | right type | "
         "fixed (IoU) | fixed by addition (IoU) | false adds | gap | tol. mIoU before -> after | "
         "valid before -> after | s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(cases):
        s, m = c["score"], c["meta"]
        L.append(f"| {i + 1} | {c['pid']} | {c['kind']} ({c['typ']}) | {m['location']} | {m['position']} | "
                 f"{m['size_bin']} ({m['size_pct']}%) | {s['flags']} | {yn(s['found'])} | {yn(s['right_type'])} | "
                 f"{yn(s['fixed'])} ({s['fix_iou']:.2f}) | {yn(s['fixed_add'])} ({s['fix_iou_add']:.2f}) | "
                 f"{s['false_adds']} | {s['gap'] or '-'} | {s['tol_before']} -> {s['tol_after']} | "
                 f"{yn(s['valid_before'])} -> {yn(s['valid_after'])} | {c['seconds']:.0f} |")
    tot = lambda k: sum(bool(s[k]) for s in S)
    L += ["", f"**Total:** found {tot('found')} / {n}, right type {tot('right_type')} / {n}, fixed "
              f"{tot('fixed')} / {n} (by addition {tot('fixed_add')} / {n}), false additions "
              f"{sum(s['false_adds'] for s in S)} "
              f"(on {sum(s['false_adds'] > 0 for s in S)} drawings); tolerant mIoU "
              f"{np.mean([s['tol_before'] for s in S]):.1f} -> {np.mean([s['tol_after'] for s in S]):.1f}; validity "
              f"{100 * np.mean([s['valid_before'] for s in S]):.0f}% -> {100 * np.mean([s['valid_after'] for s in S]):.0f}%; "
              f"time {np.mean([c['seconds'] for c in cases]):.0f} s per drawing "
              f"(max {max(c['seconds'] for c in cases):.0f} s)."]
    rows = [{**{f: c["meta"][f] for f in FIELDS}, "s": c["score"]} for c in cases]
    cols = {"found": lambda rs: f"{sum(r['s']['found'] for r in rs)}",
            "fixed": lambda rs: f"{sum(r['s']['fixed'] for r in rs)}",
            "false adds": lambda rs: f"{sum(r['s']['false_adds'] for r in rs)}",
            "tol. mIoU after": lambda rs: f"{np.mean([r['s']['tol_after'] for r in rs]):.1f}"}
    L += ["", "## Breakdown by omission metadata"] + breakdowns(rows, cols)
    L += ["", "## Messages per drawing", ""]
    for i, c in enumerate(cases):
        L.append(f"{i + 1}. {c['pid']} ({c['kind']}): " + ("; ".join(c["res"]["messages"]) or "nothing added"))
    L += ["", "## Validity diagnosis: rules the fill added", "",
          "Drawings with more violations after the fill than in the parse (omission region). Per rule: parse -> "
          "final (raw hybrid output before the type restriction): hybrid = the raw output has it too, restriction "
          "= only the restricted plan has it.", ""]
    diag = [(i, c) for i, c in enumerate(cases) if c["score"]["diagnosis"]]
    L += [f"- {i + 1}. {c['pid']} ({c['kind']}): {c['score']['diagnosis']}" for i, c in diag] or ["- none"]
    (OUT / f"e2e_val{tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    keys = ["found", "right_type", "fix_iou", "fixed", "fix_iou_add", "fixed_add", "false_adds", "tol_before",
            "tol_after", "valid_before", "valid_after", "viol_gt", "viol_before", "viol_after", "gap", "diagnosis",
            "flags"]
    with open(OUT / f"e2e_val{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pid", "kind", "expected", *FIELDS, "size_pct", "center_dist", *keys, "seconds"])
        for c in cases:
            m = c["meta"]
            w.writerow([c["pid"], c["kind"], c["typ"], *[m[k] for k in FIELDS], m["size_pct"], m["center_dist"],
                        *[c["score"][k] for k in keys], c["seconds"]])
    print("\n".join(L))


def compare(old_csv, new_csv, tag):
    """Old vs new run on the same drawings (old 'fixed' = by addition, the only definition then)."""
    rd = lambda f: list(csv.DictReader(open(f, encoding="utf-8")))
    old, new = rd(old_csv), rd(new_csv)
    yn = lambda v: "yes" if v == "True" else "no"
    L = ["", "## Old vs new (same 12 drawings)", "",
         "Old = before the 2026-10-20 fixes (eta 0.1; its 'fixed' counted only the fill's additions). New = eta 0, "
         "symbol-already-in-parse, door at the flag point, wall-gap rule, one room type, door + adjacent walls.", "",
         "| # | drawing | omission | old flags | new flags | old fixed by addition | new fixed by addition | "
         "new fixed | old -> new false adds | old -> new tol. mIoU after | old -> new valid after |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, (o, n) in enumerate(zip(old, new)):
        assert o["pid"] == n["pid"], (o["pid"], n["pid"])
        L.append(f"| {i + 1} | {n['pid']} | {n['kind']} | {o['flags']} | {n['flags']} | {yn(o['fixed'])} | "
                 f"{yn(n['fixed_add'])} | {yn(n['fixed'])} | {o['false_adds']} -> {n['false_adds']} | "
                 f"{o['tol_after']} -> {n['tol_after']} | {yn(o['valid_after'])} -> {yn(n['valid_after'])} |")
    cnt = lambda rows, k: sum(r[k] == "True" for r in rows)
    mean = lambda rows, k: np.mean([float(r[k]) for r in rows])
    L += ["", "| totals | found | right type | fixed (new definition) | fixed by addition | false adds | "
              "tol. mIoU after | validity after | s per drawing |", "|---|---|---|---|---|---|---|---|---|",
          f"| old | {cnt(old, 'found')} | {cnt(old, 'right_type')} | - | {cnt(old, 'fixed')} | "
          f"{sum(int(r['false_adds']) for r in old)} | {mean(old, 'tol_after'):.1f} | "
          f"{100 * cnt(old, 'valid_after') / len(old):.0f}% | {mean(old, 'seconds'):.0f} |",
          f"| new | {cnt(new, 'found')} | {cnt(new, 'right_type')} | {cnt(new, 'fixed')} | {cnt(new, 'fixed_add')} | "
          f"{sum(int(r['false_adds']) for r in new)} | {mean(new, 'tol_after'):.1f} | "
          f"{100 * cnt(new, 'valid_after') / len(new):.0f}% | {mean(new, 'seconds'):.0f} |"]
    with open(OUT / f"e2e_val{tag}.md", "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def figure(cases, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(len(cases), 4, figsize=(22, 5.5 * len(cases)), squeeze=False)
    for i, c in enumerate(cases):
        a, b, r, g = axes[i]
        s, res = c["score"], c["res"]
        a.imshow(c["img"], cmap="gray", vmin=0, vmax=255)
        ys, xs = np.nonzero(c["removed"])
        a.add_patch(Rectangle((xs.min() - 4, ys.min() - 4), xs.max() - xs.min() + 8, ys.max() - ys.min() + 8,
                              fill=False, ec="orange", lw=1.5))
        m = c["meta"]
        a.set_title(f"{i + 1}. {c['pid']}: {c['kind']} removed (orange)\n{m['location']}, {m['position']}, "
                    f"{m['size_bin']}", fontsize=9)
        b.imshow(C16.colourise(res["parsed"]), interpolation="nearest")
        for h in res["holes"]:
            y0, x0, y1, x1 = h["box"]
            b.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="red", lw=1.2))
            b.text(x0, y0 - 2, f"{h['type']} ({'+'.join(h['rules'])})", color="red", fontsize=7)
        b.contour(s["rm"], levels=[0.5], colors="orange", linewidths=0.8)
        b.set_title(f"parse + flags ({len(res['holes'])}); found {'yes' if s['found'] else 'no'}", fontsize=9)
        r.imshow(res["overlay"])
        r.set_title(f"result: fixed {'yes' if s['fixed'] else 'no'} (IoU {s['fix_iou']:.2f}; by addition "
                    f"{s['fix_iou_add']:.2f}), "
                    f"false adds {s['false_adds']}", fontsize=9)
        r.text(0, res["overlay"].shape[0] * 1.02, "\n".join(res["messages"][:5]), fontsize=8, va="top")
        g.imshow(C16.colourise(s["gt"]), interpolation="nearest")
        g.contour(s["rm"], levels=[0.5], colors="orange", linewidths=0.8)
        g.set_title(f"ground truth (omitted component in orange); tol. mIoU {s['tol_before']} -> {s['tol_after']}",
                    fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / f"e2e_val{tag}.png", dpi=55)


if __name__ == "__main__":
    main()
