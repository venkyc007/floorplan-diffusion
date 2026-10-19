"""Generous flagger + judge (user, overnight 2026-10-19/20).

    python -m src.stage2.detect.judge tune --zip parser_val.zip --tuning tuning_ids.txt     -> configs/stage2/judge.yaml
    python -m src.stage2.detect.judge eval --completion completion_val.zip --drawings parser_val.zip
        -> outputs/stage2/judge_eval.md, outputs/stage2/judge_scores.png

1. Generous CHECK: every source loosened (detector v2 + v3, C8, R5, ink tier 1, R6; railings manual-only, ink tier 2
   off) to ~3x the current false-alarm level: on the TUNING drawings the setting of the grid below with the highest
   mean recall among those with 1.2-2.0 false alarms per intact drawing (else the one closest to 1.5).
2. Judge each flag with the 16-class U-Net inpainter (k2a, unforced, NO sketch prior): a 128 x 128 window around the
   flag, the flag's box masked; judge = mean U-Net probability of the flagged type inside the box minus that type's
   share in the current parse inside the box (room flags: the wall class). Door flags get the door-ink value as a
   second feature (lowest ink score of the parsed doors in the box, capped at 10, divided by 10; 1 if none).
3. A logistic model on [judge, ink feature, is_door] is fitted on the tuning flags (true = overlaps an omission);
   flags are kept above a probability threshold chosen on the tuning drawings for <= 0.4 false alarms per intact
   drawing (the budget of the current final setting), highest mean recall.
4. eval: held-out drawings (image_eval's cases): the current final CHECK (ink tier 2 off) vs generous + judge,
   per type, mean recall, false alarms; histograms of the logistic score for true omissions vs false flags per type.
Everything is scored from typed proposals stored once per drawing (collect).
"""

import argparse
import itertools
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.stage2.complete.sketch_prior import SIZES
from src.stage2.data.classes16 import C16
from src.stage2.data.erasure import grown
from src.stage2.detect import rules as R
from src.stage2.detect.detection_eval import build_heldout_cases, detector_probs, proposals
from src.stage2.detect.detector_v3 import v3_probs
from src.stage2.detect.door_ink import category, door_ink_scores, guarded
from src.stage2.detect.gaps import closed_pairs, gap_candidates, instance_areas
from src.stage2.detect.image_eval import KINDS, omit
from src.stage2.detect.window_runs import window_candidates
from src.stage2.pipeline.run import cfg, models, parse, window_around
from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "stage2"
CHOICE = PROJECT_ROOT / "configs" / "stage2" / "detector_threshold.yaml"
JUDGE = PROJECT_ROOT / "configs" / "stage2" / "judge.yaml"
WR = yaml.safe_load((PROJECT_ROOT / "configs" / "stage2" / "window_rule.yaml").read_text(encoding="utf-8"))
I = C16.index
DET_TB = [(0.3, 10), (0.3, 20), (0.5, 10), (0.5, 20), (0.5, 40), (0.7, 20)]      # detector grid (v2 and v3)
GRID = {"det": [(0.5, 40), (0.3, 20), (0.3, 10)], "det3": [(0.5, 20), (0.3, 10)], "c8": [700, 200, 0],
        "r5": [True], "ink": [1.5, 3.0], "r6": [1.2, 1.0]}
FLAG_CLS = {"door": [I["door"]], "window": [I["window"]], "wall": [I["wall"]], "room": [I["wall"]],
            **{f: [I[f]] for f in C16.cfg["groups"]["F"]}}


def read(z, name, flag):
    return cv2.imdecode(np.frombuffer(z.read(name), np.uint8), flag)


# ------------------------------------------------------------------ judge features

@torch.no_grad()
def judge_scores(label, flags):
    """U-Net judge for each (box, type): mean P(type) in the masked box - parse share of the type in the box."""
    if not flags:
        return []
    m = models()["unet"]
    out = []
    for box, typ in flags:
        win, _, hole = window_around(label, box, 128)
        t = torch.from_numpy(win.astype(np.int64))[None]
        oh = F.one_hot(t, C16.n).permute(0, 3, 1, 2).float()
        mask = torch.from_numpy(hole).float()[None, None]
        P = F.softmax(m(torch.cat([oh * (1 - mask), mask], 1)).float(), 1)[0].numpy()
        cls = FLAG_CLS.get(typ, [I["wall"]])
        h = hole > 0
        if not h.any():
            out.append(0.0)
            continue
        out.append(float(P[cls].sum(0)[h].mean() - np.isin(win[h], cls).mean()))
    return out


def ink_feature(ink, box):
    y0, x0, y1, x1 = box
    s = [v for (b, v, _) in ink if not (b[2] < y0 or b[0] > y1 or b[3] < x0 or b[1] > x1)]
    return min(s) if s else 10.0


# ------------------------------------------------------------------ typed proposals per drawing

def collect(img, choice):
    """{shape, cands: [(source, key, box, type)], judge: {(box, type): score}, ink: [(box, score, category)]}."""
    label, _, _ = parse(img, cfg()["parse"])
    m = models()
    skip = set(choice.get("manual_only_types", []))
    p2 = detector_probs(m["detector"], label, m["detector_K"])
    p3 = v3_probs(m["detector_v3"], img, label)
    cands = []
    for tb in DET_TB:
        cands += [("det", tb, p.box, p.type) for p in proposals(p2, *tb) if p.type not in skip]
        cands += [("det3", tb, p.box, p.type) for p in proposals(p3, *tb) if p.type not in skip]
    cands += [("c8", c.score, c.box, "door") for c in R.discrete_candidates(label) if c.rules[0] == "C8"]
    g = choice.get("gap_rule", {})
    cands += [("r5", True, c.box, "door") for c in gap_candidates(label, open_rate=g.get("open_rate", 0.10),
              need_area=g.get("need_area"), small_other=g.get("small_other"), min_len=g.get("min_len"))]
    bad, area = closed_pairs(open_rate=g.get("open_rate", 0.10)), instance_areas(label)
    ink = [(R.box_of(d, label.shape), s, category(label, d, g, bad, area)) for d, s in door_ink_scores(img, label)]
    cands += [("ink", (s, cat), b, "door") for b, s, cat in ink]
    cands += [("r6", c.score, c.box, "window") for c in window_candidates(label, WR["thresholds"], WR["need_area"],
                                                                          SIZES["window"][1], 1.0)]
    uniq = list(dict.fromkeys((tuple(c[2]), c[3]) for c in cands))
    judge = dict(zip(uniq, judge_scores(label, uniq)))
    return {"shape": label.shape, "cands": cands, "judge": judge, "ink": ink}


def selected(rec, S):
    """Candidates (box, type) a setting S = {det, det3, c8, r5, ink, r6} raises on one drawing (ink tier 2 off)."""
    guard = guarded([s for _, s, _ in rec["ink"]], S["ink"]) if S["ink"] else True
    out = []
    for src, key, box, typ in rec["cands"]:
        if (src == "det" and key == S["det"]) or (src == "det3" and key == S["det3"]) or \
                (src == "c8" and key >= S["c8"]) or (src == "r5" and S["r5"]) or \
                (src == "ink" and not guard and key[1] == "relevant" and key[0] < S["ink"]) or \
                (src == "r6" and key >= S["r6"]):
            out.append((tuple(box), typ))
    return list(dict.fromkeys(out))


def features(rec, flag):
    box, typ = flag
    door = typ == "door"
    return np.array([rec["judge"][flag], min(ink_feature(rec["ink"], box), 10.0) / 10 if door else 0.0, float(door)])


def prob(coef, x):
    return 1 / (1 + np.exp(-(coef[0] + x @ np.asarray(coef[1:]))))


def fit_logistic(X, y, steps=3000, lr=0.5, l2=1e-3):
    w = np.zeros(X.shape[1] + 1)
    Xb = np.hstack([np.ones((len(X), 1)), X])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Xb @ w))
        w -= lr * (Xb.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]])
    return w.tolist()


def judged_flags(label, image, choice):
    """Pipeline use (detector_threshold.yaml use_judge: true): the generous setting's flags, kept when the judge's
    logistic score >= tau (configs/stage2/judge.yaml), merged into holes."""
    from src.stage2.pipeline.run import detect_parts
    J = yaml.safe_load(JUDGE.read_text(encoding="utf-8"))
    G = J["generous"]
    gen = dict(choice, threshold=G["det"][0], min_blob=G["det"][1], threshold_v3=G["det3"][0], min_blob_v3=G["det3"][1],
               c8_min_area=G["c8"], ink_threshold=G["ink"], ink_threshold_interior=0.0, r6_scale=G["r6"],
               detector_model="both", extra_rules=["C8", "R5", "INK", "R6"])
    flags = [p for v in detect_parts(label, image, gen).values() for p in v]
    if not flags:
        return []
    g = choice.get("gap_rule", {})
    bad, area = closed_pairs(open_rate=g.get("open_rate", 0.10)), instance_areas(label)
    ink = [(R.box_of(d, label.shape), s, category(label, d, g, bad, area)) for d, s in door_ink_scores(image, label)]
    js = judge_scores(label, [(tuple(p.box), p.type) for p in flags])
    rec = {"judge": {(tuple(p.box), p.type): j for p, j in zip(flags, js)}, "ink": ink}
    kept = [p for p in flags if prob(J["coef"], features(rec, (tuple(p.box), p.type))[None])[0] >= J["tau"]]
    return R.merge(kept)


# ------------------------------------------------------------------ scoring

def hit(flags, rm):
    return any(rm[b[0]:b[2] + 1, b[1]:b[3] + 1].any() for b, _ in flags)


def n_holes(flags):
    return len(R.merge([R.Candidate(t, b) for b, t in flags]))


def score(intact, omitted, pick):
    """pick(rec) -> kept flags. -> fa per intact drawing, per-kind recall, mean recall."""
    fa = float(np.mean([n_holes(pick(r)) for r in intact]))
    per = {k: [] for k in KINDS}
    for kd, rm, r in omitted:
        per[kd].append(hit(pick(r), rm))
    rec = {k: float(np.mean(v)) for k, v in per.items() if v}
    return {"fa": fa, "mean": float(np.mean(list(rec.values()))), **rec}


def labelled_flags(intact, omitted, S):
    """[(features, true?, type)] for every flag a setting raises (true = overlaps the omission)."""
    rows = []
    for r in intact:
        rows += [(features(r, f), 0, f[1]) for f in selected(r, S)]
    for _, rm, r in omitted:
        rows += [(features(r, f), int(hit([f], rm)), f[1]) for f in selected(r, S)]
    return rows


def fmt(name, r):
    return f"| {name} | {r['fa']:.2f} | {100 * r['mean']:.0f}% | " + " | ".join(
        f"{100 * r[k]:.0f}%" if k in r else "-" for k in KINDS) + " |"


HDR = ["| CHECK | false alarms / intact | mean recall | " + " | ".join(KINDS) + " |", "|---|---|---|" + "---|" * len(KINDS)]


def final_setting(choice):
    return {"det": (choice["threshold"], choice["min_blob"]), "det3": (choice["threshold_v3"], choice["min_blob_v3"]),
            "c8": choice.get("c8_min_area") or np.inf, "r5": "R5" in choice.get("extra_rules", []),
            "ink": choice.get("ink_threshold", 0) if "INK" in choice.get("extra_rules", []) else 0,
            "r6": (choice.get("r6_scale") or np.inf) if "R6" in choice.get("extra_rules", []) else np.inf}


# ------------------------------------------------------------------ builders

def build_tuning(zip_path, ids, n_intact, n_omitted, choice, seed=11):
    keep = set(Path(ids).read_text(encoding="utf-8").split())
    z = zipfile.ZipFile(zip_path)
    names = sorted(n[:-8] for n in z.namelist() if n.endswith("_img.png") and n[:-8] in keep)
    rng = np.random.default_rng(seed)
    intact, omitted, t0 = [], [], time.time()
    for n in names:
        if len(intact) >= n_intact and len(omitted) >= 2 * n_omitted:
            break
        lbl = read(z, f"{n}_lbl.png", cv2.IMREAD_UNCHANGED)
        if (lbl == 255).any():
            continue
        img = read(z, f"{n}_img.png", cv2.IMREAD_GRAYSCALE)
        if len(intact) < n_intact:
            intact.append(collect(img, choice))
        else:
            j = len(omitted) // 2
            for k in (2 * j, 2 * j + 1):
                for step in range(len(KINDS)):
                    o = omit(KINDS[(k + step) % len(KINDS)], img, lbl, rng)
                    if o is not None:
                        rec = collect(o[0], choice)
                        rm = cv2.resize(o[1].astype(np.uint8), rec["shape"][::-1], interpolation=cv2.INTER_NEAREST) > 0
                        omitted.append((KINDS[(k + step) % len(KINDS)], grown(rm, 3), rec))
                        break
        print(f"intact {len(intact)}, omissions {len(omitted)} ({time.time() - t0:.0f} s)", flush=True)
    return intact, omitted


def build_heldout(completion, drawings, choice, limit=None):
    """Exactly image_eval's held-out cases (59 intact + 100 drawings x 2 omissions, seed 5)."""
    held, _ = build_heldout_cases(completion)
    ids_i = [c["pid"] for c in held if c["mask"] is None]
    ids_o = list(dict.fromkeys(c["pid"] for c in held if c["mask"] is not None))
    del held
    if limit:
        ids_i, ids_o = ids_i[:limit], ids_o[:limit]
    z = zipfile.ZipFile(drawings)
    rng = np.random.default_rng(5)
    intact = [collect(read(z, f"{p}_img.png", cv2.IMREAD_GRAYSCALE), choice) for p in ids_i]
    omitted, t0 = [], time.time()
    for j, pid in enumerate(ids_o):
        img = read(z, f"{pid}_img.png", cv2.IMREAD_GRAYSCALE)
        lbl = read(z, f"{pid}_lbl.png", cv2.IMREAD_UNCHANGED)
        for k in (2 * j, 2 * j + 1):
            for step in range(len(KINDS)):
                kind = KINDS[(k + step) % len(KINDS)]
                o = omit(kind, img, lbl, rng)
                if o is not None:
                    rec = collect(o[0], choice)
                    rm = cv2.resize(o[1].astype(np.uint8), rec["shape"][::-1], interpolation=cv2.INTER_NEAREST) > 0
                    omitted.append((kind, grown(rm, 3), rec))
                    break
        print(f"held-out omitted {j + 1}/{len(ids_o)} ({time.time() - t0:.0f} s)", flush=True)
    return intact, omitted


# ------------------------------------------------------------------ main

def tune(args):
    choice = yaml.safe_load(CHOICE.read_text(encoding="utf-8"))
    intact, omitted = build_tuning(args.zip, args.tuning, args.intact, args.omitted, choice)
    final = final_setting(choice)
    base = score(intact, omitted, lambda r: selected(r, final))
    grid = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    loose = [(S, score(intact, omitted, lambda r, S=S: selected(r, S))) for S in grid]
    band = [x for x in loose if args.lo <= x[1]["fa"] <= args.hi]
    S, gen = max(band, key=lambda x: x[1]["mean"]) if band else min(loose, key=lambda x: abs(x[1]["fa"] - 1.5))
    rows = labelled_flags(intact, omitted, S)
    X = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows], float)
    coef = fit_logistic(X, y)
    best = None
    for tau in np.round(np.linspace(0.02, 0.98, 49), 2):
        pick = lambda r, tau=tau: [f for f in selected(r, S) if prob(coef, features(r, f)[None])[0] >= tau]
        res = score(intact, omitted, pick)
        if res["fa"] <= args.budget and (best is None or res["mean"] > best[1]["mean"]):
            best = (float(tau), res)
    tau, judged = best if best else (0.98, None)
    out = {"generous": {k: (list(v) if isinstance(v, tuple) else v) for k, v in S.items()},
           "coef": coef, "features": ["judge", "ink / 10 (doors; 1 = no door)", "is_door"], "tau": tau,
           "source": f"src/stage2/detect/judge.py tune: {len(intact)} intact + {len(omitted)} omissions on tuning "
                     f"drawings, generous band {args.lo}-{args.hi}, budget {args.budget}"}
    JUDGE.write_text("# GENERATED by python -m src.stage2.detect.judge tune\n" + yaml.safe_dump(out, sort_keys=False),
                     encoding="utf-8")
    L = ["# Generous flagger + judge: tuning (parsed TUNING drawings)", "",
         f"{len(intact)} intact + {len(omitted)} omissions. Generous setting = best mean recall with {args.lo}-{args.hi} "
         f"false alarms per intact drawing; judge threshold for <= {args.budget}.", "",
         f"Generous setting: {S}. Logistic coefficients (intercept, judge, ink, is_door): "
         f"{[round(c, 3) for c in coef]}; threshold {tau}. Flags used for the fit: {len(rows)} ({int(y.sum())} true).", ""]
    L += HDR + [fmt("current final (ink tier 2 off)", base), fmt("generous (no judge)", gen)]
    if judged:
        L.append(fmt("generous + judge", judged))
    (OUT / "judge_tune.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def evaluate(args):
    choice = yaml.safe_load(CHOICE.read_text(encoding="utf-8"))
    J = yaml.safe_load(JUDGE.read_text(encoding="utf-8"))
    S = {k: (tuple(v) if isinstance(v, list) else v) for k, v in J["generous"].items()}
    coef, tau = J["coef"], J["tau"]
    intact, omitted = build_heldout(args.completion, args.drawings, choice, args.limit)
    final = final_setting(choice)
    res_final = score(intact, omitted, lambda r: selected(r, final))
    res_gen = score(intact, omitted, lambda r: selected(r, S))
    judged = lambda r: [f for f in selected(r, S) if prob(coef, features(r, f)[None])[0] >= tau]
    res_j = score(intact, omitted, judged)
    win = res_j["mean"] > res_final["mean"] and res_j["fa"] <= res_final["fa"] + 0.1
    L = ["# Generous flagger + judge vs the current final CHECK (held-out drawings)", "",
         f"{len(intact)} intact + {len(omitted)} omissions (image_eval's held-out cases). Judge settings: "
         "configs/stage2/judge.yaml (tuned on tuning drawings only).", ""] + HDR + \
        [fmt("current final (ink tier 2 off)", res_final), fmt("generous (no judge)", res_gen),
         fmt("generous + judge", res_j), "",
         f"**Verdict: {'generous + judge WINS' if win else 'the current final stays'}** (rule: higher held-out mean "
         "recall at a false-alarm level no more than 0.1 above the current final)."]
    (OUT / "judge_eval.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    # score histograms: true omissions vs false flags, per type
    rows = labelled_flags(intact, omitted, S)
    types = sorted({t for _, _, t in rows})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(types), figsize=(3.2 * len(types), 3), squeeze=False)
    for t, ax in zip(types, axes[0]):
        p = [prob(coef, x[None])[0] for x, y, tt in rows if tt == t and y]
        q = [prob(coef, x[None])[0] for x, y, tt in rows if tt == t and not y]
        ax.hist([q, p], bins=np.linspace(0, 1, 21), label=["false flags", "true omissions"], color=["grey", "tab:red"])
        ax.axvline(tau, color="k", ls="--", lw=0.8)
        ax.set_title(f"{t} ({len(p)} true / {len(q)} false)", fontsize=8)
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Judge score (logistic) of the generous flags, held-out drawings; dashed = threshold", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "judge_scores.png", dpi=90)
    with open(OUT / "judge_eval.md", "a", encoding="utf-8") as f:
        f.write(f"\nVERDICT_WIN={win}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["tune", "eval"])
    ap.add_argument("--zip")
    ap.add_argument("--tuning")
    ap.add_argument("--completion")
    ap.add_argument("--drawings")
    ap.add_argument("--intact", type=int, default=40)
    ap.add_argument("--omitted", type=int, default=80)
    ap.add_argument("--lo", type=float, default=1.2)
    ap.add_argument("--hi", type=float, default=2.0)
    ap.add_argument("--budget", type=float, default=0.4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    tune(args) if args.mode == "tune" else evaluate(args)


if __name__ == "__main__":
    main()
