"""Large omissions: compare the three large-hole fills (user, 2026-10-20). Kaggle CPU kernel.

    python -m src.stage2.pipeline.large_fill_compare --zip parser_val.zip [--limit N] [--mode both|manual|auto]
        -> outputs/stage2/large_fill_compare.md / .csv / .png

Cases (all held-out val drawings, fixed order as in e2e_val, nothing picked by hand):
  batch 3   its 8 large-kind cases (4 room (whole), 4 region; e2e_val --batch 3, seed 22)
  medium    6 NEW drawings (batches 1-3 left out), omission measured in the medium bin (2.7-7.7% of the building),
            src/stage2/detect/omission_mix.omit_in_bin("medium") (room (whole) or region, size range cut to the bin)
  large     6 NEW drawings, omit_in_bin("large"); seed 23 for both
Strategies (src/stage2/complete/large_fill.py): B1 big window, B2 coarse-to-fine with refinement t0 150 and 300,
B3 tiled. Fast sampling (configs/stage2/pipeline.yaml `fast`), eta 0.
Modes
  manual    the hole is GIVEN: the box of the true omission area (the room + its walls / the rectangle) at the
            parsed map's scale; fill quality without detection
  auto      the whole pipeline (parse -> CHECK with R8 "region missing" -> fill; region flags use the strategy,
            other flags the usual small-hole fill)
Scores, inside the omission area's box (completion scale; ground truth resized nearest):
  found     (auto) a flag overlaps the erased components (grown 3 px); "region flag" = such a flag is R8's region type
  fixed     found (manual: always) AND wall IoU >= 0.3 between the final plan's wall pixels within 3 px of the erased
            components and those components (as for room omissions, e2e_val.right_class_iou, local)
  tol. mIoU / validity   hole_ious / thresholded violations (on a window covering the box) of the final plan vs the
            ground truth; valid = no more violations than the ground truth (Stage 1 rule); also the parse before
  time      manual: seconds per hole (the fill only); auto: seconds per drawing (whole pipeline)
"""

import argparse
import csv
import time
import zipfile

import cv2
import numpy as np
import torch

from src.stage2.complete.large_fill import fill_large
from src.stage2.complete.t0_sweep import hole_ious
from src.stage2.data.classes16 import C16
from src.stage2.data.erasure import grown
from src.stage2.detect.image_eval import LARGE_KINDS
from src.stage2.detect.omission_mix import omit_in_bin
from src.stage2.pipeline import e2e_val
from src.stage2.pipeline.run import cfg, generator, parse, run_pipeline
from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "stage2"
I = C16.index
STRATS = [("B1 big window", "big", None), ("B2 coarse, t0 150", "coarse", 150), ("B2 coarse, t0 300", "coarse", 300),
          ("B3 tiled", "tiled", None)]
N_NEW = 6


def read(z, name, flag):
    return cv2.imdecode(np.frombuffer(z.read(name), np.uint8), flag)


def cases(z, limit=None):
    ns = argparse.Namespace(limit=None)
    first = {c["pid"] for c in e2e_val.pick(z, ns)}
    second = {c["pid"] for c in e2e_val.pick(z, ns, exclude=first, seed=21)}
    b3 = e2e_val.pick(z, ns, exclude=first | second, seed=22, plan=e2e_val.BATCH3)
    used = first | second | {c["pid"] for c in b3}
    out = [dict(c, group="batch 3") for c in b3 if c["kind"] in LARGE_KINDS]
    held = e2e_val.HELDOUT.read_text(encoding="utf-8").split()
    order = [held[i] for i in np.random.default_rng(20).permutation(len(held))]
    order = [p for p in order if p not in e2e_val.CHECK_SET and p not in used]
    rng = np.random.default_rng(23)
    for target in ["medium"] * N_NEW + ["large"] * N_NEW:
        for pid in order:
            if pid in used:
                continue
            lbl = read(z, f"{pid}_lbl.png", cv2.IMREAD_UNCHANGED)
            if (lbl == 255).any():
                continue
            img = read(z, f"{pid}_img.png", cv2.IMREAD_GRAYSCALE)
            o = omit_in_bin(target, img, lbl, rng)
            if o is None:
                continue
            used.add(pid)
            out.append({"pid": pid, "kind": o[4]["type"], "img": o[0], "removed": o[1], "typ": o[2], "meta": o[4],
                        "lbl": lbl, "area": o[5], "group": f"new {target}"})
            break
    return out[:limit] if limit else out


def to_parse_scale(m, shape):
    return cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def box_of(m):
    ys, xs = np.nonzero(m)
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def violations(plan, box):
    """Thresholded violations inside box, on a window (multiple of 32, >= 128) that covers it + 32 px."""
    side = max(128, int(np.ceil((max(box[2] - box[0], box[3] - box[1]) + 65) / 32) * 32))
    return e2e_val.violations(plan, box, win=side)


def score(plan, parsed, gt, rm, box, holes=None):
    rmg = grown(rm, 3)
    reg = np.zeros(rm.shape, bool)
    reg[box[0]:box[2] + 1, box[1]:box[3] + 1] = True
    s = {}
    if holes is not None:
        over = [h for h in holes if rmg[h["box"][0]:h["box"][2] + 1, h["box"][1]:h["box"][3] + 1].any()]
        s.update(found=bool(over), region_flag=any(h["type"] == "region" for h in over),
                 flags="; ".join(f"{h['type']} ({'+'.join(h['rules'])})" for h in holes) or "none")
    found = s.get("found", True)
    iou = e2e_val.right_class_iou(plan, I["wall"], rm, rmg, local=True)
    vg = violations(gt, box)
    s.update(fix_iou=round(iou, 3), fixed=found and iou >= e2e_val.FIX_IOU,
             tol=round(100 * hole_ious(plan, gt, reg)[1], 1), tol_parse=round(100 * hole_ious(parsed, gt, reg)[1], 1),
             valid=violations(plan, box) <= vg, valid_parse=violations(parsed, box) <= vg)
    return s


def run_case(c, mode):
    c["res"] = {}
    pc, fc, fast = cfg()["parse"], cfg()["fill"], cfg()["fast"]
    parsed, _, _ = parse(c["img"], pc)
    c["parsed"] = parsed
    c["gt"] = cv2.resize(c["lbl"], (parsed.shape[1], parsed.shape[0]), interpolation=cv2.INTER_NEAREST)
    c["rm"] = to_parse_scale(c["removed"], parsed.shape)
    if not c["rm"].any():
        c["rm"] = to_parse_scale(grown(c["removed"], 2), parsed.shape)
    area = to_parse_scale(c["area"], parsed.shape)
    c["box"] = box_of(area | c["rm"])
    hole = np.zeros(parsed.shape, bool)
    hole[c["box"][0]:c["box"][2] + 1, c["box"][1]:c["box"][3] + 1] = True
    for name, strat, t0r in STRATS:
        if mode in ("manual", "both"):
            g = generator(0)
            t = time.time()
            new, info = fill_large(parsed, hole, strat, fc, fast, g, t0r or 150)
            sec = time.time() - t
            c["res"][("manual", name)] = {"plan": new, "s": {**score(new, parsed, c["gt"], c["rm"], c["box"]),
                                                             "seconds": round(sec, 1)}, "info": info}
        if mode in ("auto", "both"):
            t = time.time()
            res = run_pipeline(c["img"], fast=True, large=strat, t0_refine=t0r)
            sec = time.time() - t
            c["res"][("auto", name)] = {"plan": res["completed"], "messages": res["messages"],
                                        "s": {**score(res["completed"], res["parsed"], c["gt"], c["rm"], c["box"],
                                                      res["holes"]), "seconds": round(sec, 1)}}
        print(c["pid"], c["group"], name, {m: r["s"] for (m, n), r in c["res"].items() if n == name}, flush=True)


def report(cs, mode, tag):
    yn = lambda b: "yes" if b else "no"
    modes = [m for m in ("manual", "auto") if mode in (m, "both")]
    L = [f"# Large holes: three fills compared{tag}", "",
         f"{len(cs)} held-out val drawings: " + ", ".join(f"{g} {sum(c['group'] == g for c in cs)}" for g in
                                                         ("batch 3", "new medium", "new large")) +
         ". Size bins: " + ", ".join(f"{b} {sum(c['meta']['size_bin'] == b for c in cs)}" for b in ("small", "medium", "large")) +
         ". Fast sampling, eta 0, CPU. Scores inside the omission area's box at the parsed map's scale (see the "
         "script docstring). Manual = the hole box given (fill quality alone); auto = the full pipeline with R8.", ""]
    for m in modes:
        L += [f"## {m} mode: totals", "",
              "| strategy | " + ("found | region flag | " if m == "auto" else "") +
              "fixed | tol. mIoU (parse -> fill) | valid (parse -> fill) | s per " + ("hole" if m == "manual" else "drawing") + " |",
              "|---|" + ("---|---|" if m == "auto" else "") + "---|---|---|---|"]
        for name, _, _ in STRATS:
            S = [c["res"][(m, name)]["s"] for c in cs if (m, name) in c["res"]]
            if not S:
                continue
            cnt = lambda k: f"{sum(bool(s[k]) for s in S)} / {len(S)}"
            L.append(f"| {name} | " + (f"{cnt('found')} | {cnt('region_flag')} | " if m == "auto" else "") +
                     f"{cnt('fixed')} | {np.mean([s['tol_parse'] for s in S]):.1f} -> {np.mean([s['tol'] for s in S]):.1f} | "
                     f"{100 * np.mean([s['valid_parse'] for s in S]):.0f}% -> {100 * np.mean([s['valid'] for s in S]):.0f}% | "
                     f"{np.mean([s['seconds'] for s in S]):.0f} |")
        L += ["", f"By size bin ({m}; fixed / tol. mIoU after per strategy):", "",
              "| size bin | n | " + " | ".join(n for n, _, _ in STRATS) + " |", "|---|---|" + "---|" * len(STRATS)]
        for b in ("medium", "large"):
            sub = [c for c in cs if c["meta"]["size_bin"] == b]
            if sub:
                L.append(f"| {b} | {len(sub)} | " + " | ".join(
                    f"{sum(c['res'][(m, n)]['s']['fixed'] for c in sub)} / {np.mean([c['res'][(m, n)]['s']['tol'] for c in sub]):.1f}"
                    for n, _, _ in STRATS) + " |")
        L += ["", f"Per case ({m}): fixed (wall IoU) / tol. mIoU per strategy", "",
              "| # | drawing | group | omission | size | " + ("flags | " if m == "auto" else "") + "parse tol. | " +
              " | ".join(n for n, _, _ in STRATS) + " |",
              "|---|---|---|---|---|" + ("---|" if m == "auto" else "") + "---|" + "---|" * len(STRATS)]
        for i, c in enumerate(cs):
            first = c["res"][(m, STRATS[0][0])]["s"]
            L.append(f"| {i + 1} | {c['pid']} | {c['group']} | {c['kind']} | {c['meta']['size_bin']} ({c['meta']['size_pct']}%) | " +
                     (f"{first['flags']} | " if m == "auto" else "") + f"{first['tol_parse']} | " +
                     " | ".join(f"{yn(c['res'][(m, n)]['s']['fixed'])} ({c['res'][(m, n)]['s']['fix_iou']:.2f}) / "
                                f"{c['res'][(m, n)]['s']['tol']}" for n, _, _ in STRATS) + " |")
        L.append("")
    (OUT / f"large_fill_compare{tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    with open(OUT / f"large_fill_compare{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pid", "group", "kind", "size_bin", "size_pct", "mode", "strategy", "found", "region_flag", "fixed",
                    "fix_iou", "tol_parse", "tol", "valid_parse", "valid", "seconds"])
        for c in cs:
            for (m, n), r in c["res"].items():
                s = r["s"]
                w.writerow([c["pid"], c["group"], c["kind"], c["meta"]["size_bin"], c["meta"]["size_pct"], m, n,
                            s.get("found", ""), s.get("region_flag", ""), s["fixed"], s["fix_iou"], s["tol_parse"],
                            s["tol"], s["valid_parse"], s["valid"], s["seconds"]])
    print("\n".join(L))


def figure(cs, tag, mode="manual"):
    if len(STRATS) < 4:
        return
    """6 cases (the first 2 of each group, no selection): input | B1 | B2 (t0 of the better manual mean) | B3 | GT."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b2 = max(("B2 coarse, t0 150", "B2 coarse, t0 300"),
             key=lambda n: np.mean([c["res"][(mode, n)]["s"]["tol"] for c in cs if (mode, n) in c["res"]]))
    pick = [c for g in ("batch 3", "new medium", "new large") for c in [c for c in cs if c["group"] == g][:2]]
    cols = [("B1 big window", "B1 big window"), (b2, b2.replace("B2 coarse", "B2 coarse-to-fine")), ("B3 tiled", "B3 tiled")]
    fig, axes = plt.subplots(len(pick), 5, figsize=(25, 5 * len(pick)), squeeze=False)
    for c, ax in zip(pick, axes):
        y0, x0, y1, x1 = c["box"]
        H, W = c["parsed"].shape
        m = 16
        sl = (slice(max(0, y0 - m), min(H, y1 + m + 1)), slice(max(0, x0 - m), min(W, x1 + m + 1)))
        fy, fx = c["img"].shape[0] / H, c["img"].shape[1] / W
        isl = (slice(int(sl[0].start * fy), int(sl[0].stop * fy)), slice(int(sl[1].start * fx), int(sl[1].stop * fx)))
        ax[0].imshow(c["img"][isl], cmap="gray", vmin=0, vmax=255)
        ax[0].set_title(f"{c['pid']}\n{c['kind']}, {c['meta']['size_bin']} ({c['meta']['size_pct']}%): input", fontsize=9)
        for a, (key, title) in zip(ax[1:4], cols):
            r = c["res"][(mode, key)]
            a.imshow(C16.colourise(r["plan"][sl]), interpolation="nearest")
            s = r["s"]
            a.set_title(f"{title}: fixed {'yes' if s['fixed'] else 'no'} (IoU {s['fix_iou']:.2f})\n"
                        f"tol. mIoU {s['tol']} (parse {s['tol_parse']}), {s['seconds']:.0f} s", fontsize=9)
        ax[4].imshow(C16.colourise(c["gt"][sl]), interpolation="nearest")
        ax[4].set_title("ground truth (orange: erased components)", fontsize=9)
        for a in ax[1:]:
            a.add_patch(plt.Rectangle((x0 - sl[1].start - .5, y0 - sl[0].start - .5), x1 - x0 + 1, y1 - y0 + 1,
                                      fill=False, ec="red", lw=1))
        ax[4].contour(c["rm"][sl], levels=[.5], colors="orange", linewidths=0.8)
        for a in ax:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / f"large_fill_compare{tag}.png", dpi=55)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mode", default="both", choices=("both", "manual", "auto"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--strats", default=None, help="comma-separated subset of B1,B2,B3 (default: all)")
    args = ap.parse_args()
    if args.strats:
        keep = args.strats.split(",")
        STRATS[:] = [s for s in STRATS if s[0].split()[0] in keep]
    cs = cases(zipfile.ZipFile(args.zip), args.limit)
    print("cases:", [(c["pid"], c["group"], c["kind"], c["meta"]["size_bin"], c["meta"]["size_pct"]) for c in cs], flush=True)
    for c in cs:
        run_case(c, args.mode)
    report(cs, args.mode, args.tag)
    if args.mode in ("both", "manual"):
        figure(cs, args.tag)


if __name__ == "__main__":
    main()
