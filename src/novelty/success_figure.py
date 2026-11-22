"""Poster figure: three SUCCESSFUL reconstructions of large omissions by the final hybrid (v4c).

    python -m src.novelty.success_figure            # runs v4c on the 3 cases (laptop CPU, ~2 min each), then draws
    python -m src.novelty.success_figure --draw     # redraw only, from the saved runs

How the cases are chosen (fixed rule, no looking at pictures; docs/decisions.md):
  the held-out large omissions in evaluation order (src/novelty/walkthrough.large_cases, same generator and seeds
  as eval_e2e), keeping the first three that v4c got fully right IN THE HELD-OUT EVALUATION (kernel v16,
  outputs/novelty/e2e_heldout_round4.csv): found AND recovered (2 px) AND fixed (strict, IoU >= 0.3) AND sound.
  The case is then re-run here and scored with the evaluation's own code (eval_e2e.row_of); a case whose laptop
  re-run is not a full success is skipped and reported (the CPU diffusion differs a little from the Kaggle GPU).
These are successes by construction, so the caption says how often a full success happens (held-out and test).

Columns: corrupted drawing (red dashes = erased area, not given to the system) | reconstruction drawn on the drawing
(red = added by v4c) | reconstruction as a 16-class map | true layout. Under each row: the case's scores.
Output: outputs/novelty/poster/qualitative_successes.png. Saved runs (CubiCasa crops, not committed):
data/novelty_success/. A viewer only: nothing is tuned here.
"""

import argparse
import csv
import pickle
import zipfile

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.novelty import pipeline as nov
from src.novelty.assessment_figures import setup
from src.novelty.eval_e2e import row_of
from src.novelty.walkthrough import large_cases
from src.stage2.data.classes16 import C16
from src.utils import PROJECT_ROOT

CONFIG = "configs/novelty/pipeline_sound_select_c.yaml"           # v4c, the final hybrid
ZIP = PROJECT_ROOT / "outputs/kaggle/stage2_data_v3/stage2_data/parser_val.zip"
HELDOUT = PROJECT_ROOT / "outputs/novelty/e2e_heldout_round4.csv"
TEST = PROJECT_ROOT / "outputs/novelty/e2e_test_all_arms.csv"
CACHE = PROJECT_ROOT / "data/novelty_success"
OUT = PROJECT_ROOT / "outputs/novelty/poster/qualitative_successes.png"
ARM = "hybrid v4c auto"
N = 3


def full_success(r):
    return all(str(r.get(k, "")) == "True" for k in ("found", "recovered", "fixed", "s_sound"))


def success_rate(path):
    R = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["arm"] == ARM and r["target"] == "large"]
    return sum(map(full_success, R)), len(R)


def run_cases():
    held = {r["pid"]: r for r in csv.DictReader(open(HELDOUT, encoding="utf-8"))
            if r["arm"] == ARM and r["target"] == "large"}
    nov.CFG_FILE = PROJECT_ROOT / CONFIG
    CACHE.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(ZIP)
    kept, skipped = [], []
    for c in large_cases(z, "heldout", 151):              # evaluation order; stops early below
        if not full_success(held.get(c["pid"], {})):
            continue
        res = nov.run(c["o"][0], seed=0)
        o = c["o"]
        row = row_of(c["pid"], "large", {0: o[0], 1: o[1], 2: o[2], 4: o[4], 5: o[5], "lbl": c["lbl"]}, "laptop", res, 0)
        print(c["pid"], {k: row.get(k) for k in ("found", "recovered", "fixed", "fix_iou", "s_sound")}, flush=True)
        if not full_success(row):
            skipped.append(c["pid"])
            continue
        keep = {"pid": c["pid"], "o": {k: o[k] for k in (0, 1, 2, 4, 5)}, "img": c["img"], "lbl": c["lbl"],
                "parsed": res["parsed"], "completed": res["completed"], "overlay": res["overlay"], "row": row}
        (CACHE / f"{len(kept)}.pkl").write_bytes(pickle.dumps(keep))
        kept.append(c["pid"])
        if len(kept) == N:
            break
    print("kept", kept, "skipped (laptop re-run not a full success)", skipped)


def yes(v):
    return "✓" if str(v) == "True" else "✗"


def draw():
    cases = [pickle.loads((CACHE / f"{k}.pkl").read_bytes()) for k in range(N)]
    heads = ["Corrupted drawing (input)\nred dashes = erased, NOT given", "Reconstruction on the drawing\n(red = added by v4c)",
             "Reconstruction (16-class map)", "True layout\n(for comparison only)"]
    fig, axs = plt.subplots(N, 4, figsize=(16, 4.1 * N + 0.6), dpi=220)
    for i, c in enumerate(cases):
        box, region, gt, sl, isl, (fy, fx) = setup(c, c["parsed"])
        def rect(f):                                      # the erased area's box; f = drawing pixels per map pixel
            return plt.Rectangle(((box[1] - sl[1].start) * f[1], (box[0] - sl[0].start) * f[0]),
                                 (box[3] - box[1] + 1) * f[1], (box[2] - box[0] + 1) * f[0], fill=False, ec="red",
                                 ls="--", lw=1.4)
        panels = [(c["o"][0][isl], "gray", (fy, fx)), (c["overlay"][isl], None, None),
                  (C16.colourise(c["completed"][sl]), None, (1, 1)), (C16.colourise(gt[sl]), None, (1, 1))]
        for j, (img, cmap, f) in enumerate(panels):
            ax = axs[i, j]
            ax.imshow(img, cmap=cmap, vmin=0, vmax=255, interpolation="nearest")
            if f is not None:                             # no box on the overlay: the red there is what was added
                ax.add_patch(rect(f))
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(heads[j], fontsize=14, fontweight="bold", pad=10)
        r = c["row"]
        kind = c["o"][4]["type"] if isinstance(c["o"][4], dict) and "type" in c["o"][4] else r.get("kind", "")
        axs[i, 0].set_ylabel(f"Case {i + 1} · {kind}, {float(r['size_pct']):.1f}%", fontsize=13, fontweight="bold")
        axs[i, 2].set_xlabel(f"found {yes(r['found'])}  recovered {yes(r['recovered'])}  fixed {yes(r['fixed'])} "
                             f"(IoU {float(r['fix_iou']):.2f})  sound {yes(r['s_sound'])}", fontsize=12,
                             color="#0b7a3e", fontweight="bold")
        axs[i, 3].set_xlabel("sound ✓" if str(r.get("st_sound")) == "True" else "true layout itself: not sound",
                             fontsize=12, color="#444")
    kh, nh = success_rate(HELDOUT)
    kt, nt = success_rate(TEST)
    fig.text(0.5, 0.004, f"Successes, chosen by a fixed rule: the first three held-out large omissions (evaluation "
             f"order) that v4c got fully right (found + recovered + fixed + sound).\nSuch full successes: "
             f"{kh} of {nh} held-out cases ({100 * kh / nh:.0f}%), {kt} of {nt} test cases ({100 * kt / nt:.0f}%). "
             "Sound = walls connect, openings in walls, rooms closed + reachable (judged without the true layout).",
             ha="center", fontsize=11.5, color="#444", linespacing=1.5)
    plt.tight_layout(rect=(0, 0.035, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, facecolor="white")
    print("->", OUT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draw", action="store_true", help="redraw from the saved runs only")
    a = ap.parse_args()
    if not a.draw:
        run_cases()
    draw()


if __name__ == "__main__":
    main()
