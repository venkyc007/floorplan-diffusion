"""Figures for the review's two picture requests (novelty round 4).

    # 1. large omissions: corrupted plan, reconstruction and violation maps together (first K cases, no picking)
    NOVELTY_CONFIG=configs/novelty/pipeline_sound.yaml python -m src.novelty.assessment_figures large \
        --zip outputs/kaggle/stage2_data_v3/stage2_data/parser_val.zip --split heldout --cases 6
    # 2. one floor plan, every use of the constraints switched on in turn
    python -m src.novelty.assessment_figures constraints --zip ... --split tuning --case 0
    # 3. plausible alternatives: one large omission filled by 4 diffusion samples, each finished by the configured
    #    version, with its violation map and the three checks, next to the true layout
    NOVELTY_CONFIG=configs/novelty/pipeline_sound.yaml python -m src.novelty.assessment_figures alternatives --zip ... --case 0

Output: outputs/novelty/assessment/ (large_*.png, constraints_*.png, alternatives_*.png).

What the violation maps show (both figures): the project's own rule check (src/stage2/pipeline/e2e_val.py, the 18
constraints of docs/constraints_spec.md; connected blobs of each rule's violation map in a window around the
omission), coloured by the group the rule belongs to:
    red     walls (C1 walls connect, C2 no floating pieces, C3 axis-aligned, C4 thickness, C5 walls continue)
    magenta openings (C9 in a wall, C10 cut through it, C11 door joins two spaces, C12 window faces out, C13 length)
    orange  rooms (C6 different spaces never touch, C7 minimum width, C8 reachable through doors, R1 area, R3 pairs)
    cyan    fixtures / railings (C14-C18)
and, outlined in black, the problems of the three structural checks of the review (src/novelty/soundness.py):
wall dead ends and floating pieces, openings not in a wall, rooms open to the outside or without a door.
Under every map: the counts, and PASS / FAIL for each of the three checks.

How it fits in: a viewer only. The pipeline version is the config in NOVELTY_CONFIG (figure 1) or the list
VERSIONS (figure 2); nothing is tuned here. Cases are the evaluation's own (src/novelty/walkthrough.large_cases).
On the laptop's CPU the diffusion can differ by a few pixels from the Kaggle GPU runs.
"""

import argparse
import zipfile
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.novelty import pipeline as nov
from src.novelty import soundness as snd
from src.novelty.eval_e2e import OUT
from src.novelty.walkthrough import crop_of, large_cases
from src.stage2.complete import sampler
from src.stage2.constraints.constraints import CONST, Ctx, one_hot, violation_map
from src.stage2.constraints import ops
from src.stage2.constraints.rules import r3_adjacency
from src.stage2.data.classes16 import C16
from src.stage2.pipeline import large_fill_compare as lfc
from src.stage2.pipeline.run import window_around
from src.utils import PROJECT_ROOT

GROUP_OF = {**{k: "walls" for k in ("C1", "C2", "C3", "C4", "C5")},
            **{k: "openings" for k in ("C9", "C10", "C11", "C12", "C13")},
            **{k: "rooms" for k in ("C6", "C7", "C8", "R1", "R3")},
            **{k: "fixtures" for k in ("C14", "C15", "C16", "C17", "C18")}}
COLOUR = {"walls": (230, 30, 30), "openings": (220, 0, 220), "rooms": (255, 150, 0), "fixtures": (0, 190, 220)}
VERSIONS = [("U-Net sketch alone\n(no constraint anywhere)", "pipeline_abl_sketch.yaml"),
            ("+ diffusion finish, 1 sample\n(model trained with the constraint losses)", "pipeline_abl_n1.yaml"),
            ("+ best of 2 by rule count\n(verifier) = v3c", "pipeline_structure_c.yaml"),
            ("+ clean-up by drawing rules\n= v3d", "pipeline_valid.yaml"),
            ("+ repair by the structural checks\n= v4", "pipeline_sound.yaml"),
            ("+ samples chosen by soundness\n= v4b", "pipeline_sound_select.yaml"),
            ("v4b without the clean-up's\nwall-end rule = v4c (final)", "pipeline_sound_select_c.yaml")]


def rule_masks(plan, box, win=128):
    """{rule: bool mask (plan size)} of the rule check's violation blobs around `box` (as e2e_val.rule_violations:
    V > 0.5 inside the box grown 7 px, in the same win x win window; R1, the room-area rule, has no map and is
    not drawn)."""
    w, (oy, ox), hole = window_around(plan, box, win)
    lab = torch.from_numpy(w.astype(np.int64))[None]
    mask = torch.from_numpy(hole).float()[None, None]
    known = torch.nn.functional.one_hot(lab, C16.n).permute(0, 3, 1, 2).float() * (1 - mask)
    P = one_hot(lab)
    M = (ops.dil(mask, 7) > 0)[0, 0].numpy()
    ctx = Ctx(hole=mask, known=known, const=CONST, K=96)
    out = {}
    for name in sampler.GUIDE:
        v = (violation_map(name, P, ctx) > 0.5)[0, 0].numpy() & M
        if v.any():
            out[name] = v
    r3 = r3_adjacency(w, CONST)[1] & M
    if r3.any():
        out["R3"] = r3
    H, W = plan.shape
    full = {}
    for k, v in out.items():
        f = np.zeros((H, W), bool)
        sy0, sx0, sy1, sx1 = max(0, oy), max(0, ox), min(H, oy + win), min(W, ox + win)
        f[sy0:sy1, sx0:sx1] = v[sy0 - oy:sy1 - oy, sx0 - ox:sx1 - ox]
        full[k] = f
    return full


def violation_image(plan, box, region):
    """The plan faded, rule blobs coloured by group, soundness problems outlined -> (rgb, text lines)."""
    im = C16.colourise(plan).astype(float)
    im = 0.35 * im + 0.65 * 255
    masks = rule_masks(plan, box)
    counts = {}
    for k, m in masks.items():
        g = GROUP_OF.get(k, "rooms")
        im[m] = COLOUR[g]
        counts[g] = counts.get(g, 0) + int(snd.ndimage.label(m, snd.K8)[1])     # blobs, as the rule check counts
    pm = snd.problem_map(plan, region)
    probs = np.zeros(plan.shape, bool)
    for k in ("dead_ends", "floating", "bad_openings", "leaks", "doorless"):
        probs |= pm[k]
    edge = probs & ~snd.ndimage.binary_erosion(probs, snd.K8)
    im[snd.ndimage.binary_dilation(edge, snd.K8)] = (0, 0, 0)
    r = snd.soundness(plan, region)
    ok = lambda v: "PASS" if v else "FAIL"
    lines = [f"rule check: {sum(counts.values())} blobs (walls {counts.get('walls', 0)}, openings "
             f"{counts.get('openings', 0)}, rooms {counts.get('rooms', 0)}, fixtures {counts.get('fixtures', 0)})",
             f"walls connect {ok(r['walls_connect'])} ({r['dead_ends']} dead ends, {r['floating']} floating)",
             f"openings in walls {ok(r['openings_in_walls'])} ({r['bad_openings']} of {r['openings']} not)",
             f"rooms closed {ok(r['rooms_closed'])} ({r['leaks']} open to outside, {r['doorless']} without door)"]
    return im.astype(np.uint8), lines, r


def legend(ax):
    ax.axis("off")
    y = 0.95
    ax.text(0.0, y, "violation map colours", fontsize=10, va="top", transform=ax.transAxes)
    for g, c in COLOUR.items():
        y -= 0.1
        ax.add_patch(plt.Rectangle((0.0, y - 0.04), 0.08, 0.06, fc=np.array(c) / 255, transform=ax.transAxes))
        ax.text(0.11, y, {"walls": "wall rules C1-C5", "openings": "opening rules C9-C13",
                          "rooms": "room rules C6-C8, R1, R3", "fixtures": "fixture / railing rules C14-C18"}[g],
                va="center", fontsize=9, transform=ax.transAxes)
    y -= 0.12
    ax.text(0.0, y, "black outline = a problem of the\nthree structural checks (soundness)", fontsize=9, va="top",
            transform=ax.transAxes)
    y -= 0.2
    for i, name in enumerate(("wall", "door", "window")):
        ax.add_patch(plt.Rectangle((0.0, y - 0.1 * i - 0.04), 0.08, 0.06, fc=C16.palette[C16.index[name]] / 255,
                                   ec="k", transform=ax.transAxes))
        ax.text(0.11, y - 0.1 * i, name, va="center", fontsize=9, transform=ax.transAxes)


def setup(c, parsed):
    H, W = parsed.shape
    o = c["o"]
    area = lfc.to_parse_scale(o[5], (H, W)) | lfc.to_parse_scale(o[1], (H, W))
    box = lfc.box_of(area)
    region = np.zeros((H, W), bool)
    region[box[0]:box[2] + 1, box[1]:box[3] + 1] = True
    gt = cv2.resize(c["lbl"], (W, H), interpolation=cv2.INTER_NEAREST)
    sl, isl, _, (fy, fx) = crop_of(area, (H, W), c["img"].shape)
    return box, region, gt, sl, isl, (fy, fx)


def outline(ax, box, sl, colour="red"):
    ax.add_patch(plt.Rectangle((box[1] - sl[1].start - .5, box[0] - sl[0].start - .5), box[3] - box[1] + 1,
                               box[2] - box[0] + 1, fill=False, ec=colour, lw=1.0, ls="--"))


def fig_large(c, res, path, version):
    parsed = res["parsed"]
    box, region, gt, sl, isl, (fy, fx) = setup(c, parsed)
    fig, ax = plt.subplots(2, 4, figsize=(20, 12))
    ax[0, 0].imshow(c["o"][0][isl], cmap="gray", vmin=0, vmax=255)
    ax[0, 0].set_title("1. Corrupted drawing given to the system\n(red dashes: what was erased - for you only)", fontsize=10)
    ax[0, 0].add_patch(plt.Rectangle(((box[1] - sl[1].start) * fx, (box[0] - sl[0].start) * fy),
                                     (box[3] - box[1] + 1) * fx, (box[2] - box[0] + 1) * fy, fill=False, ec="red",
                                     ls="--", lw=1.2))
    ax[0, 1].imshow(res["overlay"][isl])
    ax[0, 1].set_title(f"2. Reconstruction drawn on the drawing\n(red = added; {version})", fontsize=10)
    ax[0, 2].imshow(C16.colourise(res["completed"][sl]), interpolation="nearest")
    ax[0, 2].set_title("3. Reconstruction (label map)", fontsize=10)
    vi, lines, r = violation_image(res["completed"], box, region)
    ax[0, 3].imshow(vi[sl], interpolation="nearest")
    ax[0, 3].set_title("4. Violation map of the reconstruction\n" + ("SOUND" if r["sound"] else "NOT sound"), fontsize=10)
    ax[0, 3].text(0, -0.02, "\n".join(lines), transform=ax[0, 3].transAxes, fontsize=8, va="top", ha="left",
                  family="monospace", bbox=dict(fc="white", ec="none", alpha=0.85))
    ax[1, 0].imshow(C16.colourise(parsed[sl]), interpolation="nearest")
    ax[1, 0].set_title("5. Parser's map of the corrupted drawing", fontsize=10)
    vp, lp, rp = violation_image(parsed, box, region)
    ax[1, 1].imshow(vp[sl], interpolation="nearest")
    ax[1, 1].set_title("6. Violation map of the corrupted plan\n" + ("SOUND" if rp["sound"] else "NOT sound"), fontsize=10)
    ax[1, 1].text(0, -0.02, "\n".join(lp), transform=ax[1, 1].transAxes, fontsize=8, va="top", family="monospace",
                  bbox=dict(fc="white", ec="none", alpha=0.85))
    ax[1, 2].imshow(C16.colourise(gt[sl]), interpolation="nearest")
    ax[1, 2].set_title("7. True layout (one valid answer of many)", fontsize=10)
    vt, lt, rt = violation_image(gt, box, region)
    ax[1, 3].imshow(vt[sl], interpolation="nearest")
    ax[1, 3].set_title("8. Violation map of the true layout\n" + ("SOUND" if rt["sound"] else "NOT sound"), fontsize=10)
    ax[1, 3].text(0, -0.02, "\n".join(lt), transform=ax[1, 3].transAxes, fontsize=8, va="top", family="monospace",
                  bbox=dict(fc="white", ec="none", alpha=0.85))
    for a in (ax[0, 2], ax[0, 3], ax[1, 0], ax[1, 1], ax[1, 2], ax[1, 3]):
        outline(a, box, sl)
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(f"{c['pid']}: large omission ({c['o'][4]['type']}, {c['o'][4]['size_pct']}% of the building), "
                 "nothing given to the system but the corrupted drawing", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)
    return r


def fig_constraints(c, results, path):
    parsed = results[0][1]["parsed"]
    box, region, gt, sl, isl, _ = setup(c, parsed)
    n = len(results)
    fig, ax = plt.subplots(2, n + 2, figsize=(3.6 * (n + 2), 8.6))
    ax[0, 0].imshow(c["o"][0][isl], cmap="gray", vmin=0, vmax=255)
    ax[0, 0].set_title("corrupted drawing given", fontsize=9)
    vp, lp, _ = violation_image(parsed, box, region)
    ax[1, 0].imshow(vp[sl], interpolation="nearest")
    ax[1, 0].set_title("corrupted plan: violations", fontsize=9)
    ax[1, 0].text(0, -0.02, "\n".join(lp), transform=ax[1, 0].transAxes, fontsize=6.5, va="top", family="monospace")
    for j, (name, res) in enumerate(results, 1):
        ax[0, j].imshow(C16.colourise(res["completed"][sl]), interpolation="nearest")
        ax[0, j].set_title(name, fontsize=9)
        v, lines, r = violation_image(res["completed"], box, region)
        ax[1, j].imshow(v[sl], interpolation="nearest")
        ax[1, j].set_title("SOUND" if r["sound"] else "NOT sound", fontsize=9)
        ax[1, j].text(0, -0.02, "\n".join(lines), transform=ax[1, j].transAxes, fontsize=6.5, va="top",
                      family="monospace")
    ax[0, n + 1].imshow(C16.colourise(gt[sl]), interpolation="nearest")
    ax[0, n + 1].set_title("true layout", fontsize=9)
    legend(ax[1, n + 1])
    for a in ax[0, 1:].tolist() + ax[1, :n + 1].tolist():
        outline(a, box, sl)
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(f"One floor plan ({c['pid']}, large omission): what each use of the constraints changes", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def fig_alternatives(c, res, samples, path, version):
    parsed = res["parsed"]
    box, region, gt, sl, isl, _ = setup(c, parsed)
    n = len(samples)
    fig, ax = plt.subplots(2, n + 2, figsize=(3.6 * (n + 2), 8.6))
    ax[0, 0].imshow(c["o"][0][isl], cmap="gray", vmin=0, vmax=255)
    ax[0, 0].set_title("corrupted drawing given", fontsize=9)
    ax[1, 0].imshow(C16.colourise(parsed[sl]), interpolation="nearest")
    ax[1, 0].set_title("parser's map", fontsize=9)
    for j, x in enumerate(samples, 1):
        ax[0, j].imshow(C16.colourise(x[sl]), interpolation="nearest")
        ax[0, j].set_title(f"alternative {j} ({version})", fontsize=9)
        v, lines, r = violation_image(x, box, region)
        ax[1, j].imshow(v[sl], interpolation="nearest")
        ax[1, j].set_title("SOUND" if r["sound"] else "NOT sound", fontsize=9)
        ax[1, j].text(0, -0.02, "\n".join(lines), transform=ax[1, j].transAxes, fontsize=6.5, va="top",
                      family="monospace")
    ax[0, n + 1].imshow(C16.colourise(gt[sl]), interpolation="nearest")
    ax[0, n + 1].set_title("true layout (one of the valid answers)", fontsize=9)
    vt, lt, rt = violation_image(gt, box, region)
    ax[1, n + 1].imshow(vt[sl], interpolation="nearest")
    ax[1, n + 1].set_title("true layout: " + ("SOUND" if rt["sound"] else "NOT sound"), fontsize=9)
    ax[1, n + 1].text(0, -0.02, "\n".join(lt), transform=ax[1, n + 1].transAxes, fontsize=6.5, va="top",
                      family="monospace")
    for a in ax[0, 1:].tolist() + ax[1, 1:].tolist():
        outline(a, box, sl)
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(f"{c['pid']}: {n} reconstructions of one large omission (different diffusion samples, same "
                 "finishing) - several layouts can be structurally sound", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("large", "constraints", "alternatives"))
    ap.add_argument("--n", type=int, default=4, help="alternatives: diffusion samples")
    ap.add_argument("--zip", required=True)
    ap.add_argument("--split", default="heldout", choices=("heldout", "tuning"))
    ap.add_argument("--cases", type=int, default=6, help="large: the first K large cases")
    ap.add_argument("--case", type=int, default=0, help="constraints: which large case (0 = the first)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = OUT / "assessment"
    out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(a.zip)
    if a.what == "large":
        version = Path(nov.CFG_FILE).stem
        for k, c in enumerate(large_cases(z, a.split, a.cases)):
            res = nov.run(c["o"][0], seed=a.seed)
            r = fig_large(c, res, out / f"large_{a.split}_{k}_{c['pid']}.png", version)
            print(k, c["pid"], {q: r[q] for q in ("walls_connect", "openings_in_walls", "rooms_closed", "sound")}, flush=True)
    elif a.what == "alternatives":
        c = large_cases(z, a.split, a.case + 1)[a.case]
        res = nov.run(c["o"][0], seed=a.seed)
        regions = [h for h in res["holes"] if h["type"] == "region"]
        if not regions:
            print("no region flag on this drawing; nothing to show")
            return
        y0, x0, y1, x1 = regions[0]["box"]
        label = res["parsed"]
        hole = np.zeros(label.shape, bool)
        hole[y0:y1 + 1, x0:x1 + 1] = True
        cf, m = nov.cfg(), nov.models()
        nc = cf["fill"]
        g = torch.Generator().manual_seed(a.seed)
        _, info = nov.hybrid_fill(label, hole, m["unet"], m["diffusion"], nc["t0"], a.n, nc["steps_from_t0"], g, nc,
                                  all_samples=True, walls=nov.STATE.get("walls"))
        samples = [nov.finish(label, s, hole, cf)[0] for s in info["samples"]]
        fig_alternatives(c, res, samples, out / f"alternatives_{a.split}_{c['pid']}.png", Path(nov.CFG_FILE).stem)
        for j, s in enumerate(samples, 1):
            print(j, {q: v for q, v in snd.soundness(s, setup(c, label)[1]).items() if q in ("sound", "dead_ends", "doorless")})
    else:
        c = large_cases(z, a.split, a.case + 1)[a.case]
        results = []
        for name, cfg in VERSIONS:
            nov.CFG_FILE = PROJECT_ROOT / "configs" / "novelty" / cfg
            res = nov.run(c["o"][0], seed=a.seed)
            results.append((name, res))
            print(cfg, "done", flush=True)
        fig_constraints(c, results, out / f"constraints_{a.split}_{c['pid']}.png")


if __name__ == "__main__":
    main()
