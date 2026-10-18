"""Make missing-component detection usable on WHOLE plans (CHECK step), and evaluate it.

    python -m src.stage2.detect.detection_eval --zip completion_val.zip --ckpt <k2a_detector best.pt>
        -> outputs/stage2/detection_eval.md, detection_examples.png, configs/stage2/detector_threshold.yaml

Plans: typed val plans in the fixed order of rules_eval (seed 0): plans 200-299 intact (the rules'
held-out set), and 200 plans with ONE simulated omission each (seed 4, omission kinds in turn).
The learned detector runs fully convolutionally on each whole plan (padded to a multiple of 32).
Proposals from its output: missing probability = 1 - p(none); blobs = 4-connected components of
(missing probability >= threshold) with at least `min_blob` pixels; blob type = the missing class with
the highest mean probability in the blob; box = blob bounding box + 3 px.
The threshold and min_blob are swept; chosen = best mean recall over the 6 omission kinds with
<= 0.5 false alarms (proposals) per intact plan. NOTE: chosen and reported on the same 300 plans.

Detection: a proposal box overlapping the omitted pixels; type-correct: that proposal has the right type.
"both" = detector proposals + rule proposals from C8, C17, C6, C12 only (their tuned thresholds),
overlapping proposals merged (rules.merge). Rules-only = all rules with their thresholds.
"""

import argparse
from collections import defaultdict

import numpy as np
import torch
import yaml

from src.stage2.constraints.rules import components
from src.stage2.detect import rules as R
from src.stage2.detect.omissions import KINDS, MISSING, simulate_one
from src.stage2.detect.rules_eval import ordered_plans
from src.stage2.detect.train_detector import build_model, features, rule_maps
from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "stage2"
CHOICE = PROJECT_ROOT / "configs" / "stage2" / "detector_threshold.yaml"
STRONG_RULES = ("C8", "C17", "C6", "C12")
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
MIN_BLOBS = [5, 10, 20, 40, 80]


def load_detector(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_model(ck["config"]["model"])
    model.load_state_dict(ck["model"])
    return model.eval(), ck["config"]


@torch.no_grad()
def detector_probs(model, label, K=96):
    """Whole-plan detector output: (10, H, W) float16 probabilities."""
    h, w = label.shape
    H, W = -(-h // 32) * 32, -(-w // 32) * 32
    pad = np.full((H, W), 11, np.uint8)                   # background
    pad[:h, :w] = label
    x = features(torch.from_numpy(pad.astype(np.int64))[None], torch.from_numpy(rule_maps(pad))[None], K)
    return torch.softmax(model(x), 1)[0, :, :h, :w].numpy().astype(np.float16)


def proposals(probs, th, min_blob):
    miss = 1 - probs[0].astype(np.float32)
    out = []
    for m in components(miss >= th):
        if m.sum() < min_blob:
            continue
        t = 1 + int(np.argmax(probs[1:, m].astype(np.float32).mean(1)))
        out.append(R.Candidate(MISSING[t], R.box_of(m, m.shape), ["detector"], float(miss[m].max())))
    return out


def hit(cands, mask, typ):
    found = correct = False
    for c in cands:
        y0, x0, y1, x1 = c.box
        if mask[y0:y1 + 1, x0:x1 + 1].any():
            found = True
            correct |= c.type == typ
    return found, correct


def build_cases(zip_path):
    plans = ordered_plans(zip_path)
    cases = [{"pid": pid, "plan": lab, "mask": None, "kind": None, "type": None} for pid, lab in plans[200:300]]
    rng = np.random.default_rng(4)
    i = 0
    while sum(c["mask"] is not None for c in cases) < 200:
        pid, lab = plans[rng.integers(len(plans))]
        s = simulate_one(lab, rng, KINDS[i % len(KINDS)])
        i += 1
        if s is not None:
            cases.append({"pid": pid, "plan": s["plan"], "mask": s["mask"], "kind": s["kind"], "type": s["type"],
                          "orig": lab})
    return cases


def evaluate(cases, get_props):
    """-> (false alarms per intact plan, {kind: (detected, type-correct)}, mean recall over kinds)."""
    fa, n_int = 0, 0
    per = defaultdict(lambda: [0, 0, 0])
    for c in cases:
        props = get_props(c)
        if c["mask"] is None:
            fa += len(props)
            n_int += 1
        else:
            f, t = hit(props, c["mask"], c["type"])
            per[c["kind"]][0] += 1
            per[c["kind"]][1] += f
            per[c["kind"]][2] += t
    rec = {k: (v[1] / v[0], v[2] / v[0]) for k, v in per.items()}
    return fa / max(n_int, 1), rec, float(np.mean([r[0] for r in rec.values()]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    model, cfg = load_detector(args.ckpt)
    th_rules = R.load_thresholds()
    cases = build_cases(args.zip)
    for i, c in enumerate(cases):                          # cache detector output + rule candidates
        c["probs"] = detector_probs(model, c["plan"], cfg["model"]["K"])
        c["raw"] = R.raw_candidates(c["plan"])
        if (i + 1) % 25 == 0:
            print(f"cached {i + 1}/{len(cases)}", flush=True)

    rules_all = lambda c: R.detect(c["plan"], th_rules, raw=c["raw"])
    rules_strong = lambda c: R.detect(c["plan"], th_rules, raw=[r for r in c["raw"] if r.rules[0] in STRONG_RULES])
    sweep = []
    for th in THRESHOLDS:
        for mb in MIN_BLOBS:
            fa, rec, mean = evaluate(cases, lambda c: proposals(c["probs"], th, mb))
            sweep.append((th, mb, fa, mean, rec))
    ok = [s for s in sweep if s[2] <= 0.5]
    best = max(ok, key=lambda s: s[3]) if ok else min(sweep, key=lambda s: s[2])
    th, mb = best[0], best[1]
    CHOICE.write_text(yaml.safe_dump({"threshold": th, "min_blob": mb, "strong_rules": list(STRONG_RULES),
                                      "source": "detection_eval.py: best mean recall with <= 0.5 false alarms per "
                                                "intact val plan (300 val plans)"}), encoding="utf-8")
    det = lambda c: proposals(c["probs"], th, mb)
    both = lambda c: R.merge(det(c) + rules_strong(c))
    res = {name: evaluate(cases, f) for name, f in
           (("rules only (all rules)", rules_all), ("detector only", det), ("both (detector + C8/C17/C6/C12)", both))}

    L = ["# Missing-component detection on whole plans (CHECK step)", "",
         "300 typed val plans: 100 intact (val plans 200-299 of the fixed order) and 200 with ONE simulated "
         "omission each (seed 4, kinds in turn). Detected = a proposal box overlapping the omitted pixels; "
         "type-correct = that proposal has the right type. The detector setting was chosen on these same plans.", "",
         f"**Chosen detector setting: missing probability >= {th}, blobs >= {mb} px** "
         f"({best[2]:.2f} false alarms per intact plan, mean recall {100 * best[3]:.0f}%).", "",
         "| method | false alarms / intact plan | mean recall | " + " | ".join(KINDS) + " |",
         "|---|---|---|" + "---|" * len(KINDS)]
    for name, (fa, rec, mean) in res.items():
        cells = [f"{100 * rec[k][0]:.0f}% ({100 * rec[k][1]:.0f}%)" for k in KINDS]
        L.append(f"| {name} | {fa:.2f} | {100 * mean:.0f}% | " + " | ".join(cells) + " |")
    L += ["", "Cells: detected (type-correct).", "", "## Detector sweep (false alarms per intact plan / mean recall)", "",
          "| threshold | " + " | ".join(f"blob >= {m}" for m in MIN_BLOBS) + " |", "|---|" + "---|" * len(MIN_BLOBS)]
    for t in THRESHOLDS:
        row = [s for s in sweep if s[0] == t]
        L.append(f"| {t} | " + " | ".join(f"{s[2]:.2f} / {100 * s[3]:.0f}%" for s in row) + " |")
    (OUT / "detection_eval.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    figure([c for c in cases if c["mask"] is not None][:6], th, both)


def figure(cases, th, both):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from src.stage2.data.classes16 import C16
    colours = {"wall": "k", "door": "r", "window": "b", "railing": "brown", "room": "orange"}
    fig, axes = plt.subplots(6, 3, figsize=(12, 24))
    for row, c in zip(axes, cases):
        a, b, d = row
        a.imshow(C16.colourise(c["plan"]), interpolation="nearest")
        a.contour(c["mask"], levels=[0.5], colors="red", linewidths=1.2)
        a.set_title(f"{c['pid'][-24:]}: {c['kind']} omitted (red)", fontsize=8)
        b.imshow(1 - c["probs"][0].astype(np.float32), cmap="magma", vmin=0, vmax=1)
        b.set_title(f"detector: missing probability (threshold {th})", fontsize=8)
        d.imshow(C16.colourise(c["plan"]), interpolation="nearest")
        d.contour(c["mask"], levels=[0.5], colors="red", linewidths=0.8)
        for p in both(c):
            y0, x0, y1, x1 = p.box
            d.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.5, ec=colours.get(p.type, "m")))
            d.text(x0, y0 - 2, f"{p.type} ({'+'.join(p.rules)})", fontsize=6, color=colours.get(p.type, "m"))
        d.set_title("final proposals (both)", fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "detection_examples.png", dpi=80)
    print("saved", OUT / "detection_examples.png")


def build_heldout_cases(zip_path, n_intact=59):
    """Cases on val plans NOT used to choose the detector setting: the first n_intact free plans stay
    intact, every other free plan gets TWO independent omissions of different kinds (seed 5)."""
    plans = ordered_plans(zip_path)
    used = {c["pid"] for c in build_cases(zip_path)}
    free = [(p, lab) for p, lab in plans if p not in used]
    cases = [{"pid": p, "plan": lab, "mask": None, "kind": None, "type": None} for p, lab in free[:n_intact]]
    rng = np.random.default_rng(5)
    for j, (p, lab) in enumerate(free[n_intact:]):
        for kind in (KINDS[(2 * j) % len(KINDS)], KINDS[(2 * j + 1) % len(KINDS)]):
            s = simulate_one(lab, rng, kind) or simulate_one(lab, rng)
            if s is not None:
                cases.append({"pid": p, "plan": s["plan"], "mask": s["mask"], "kind": s["kind"], "type": s["type"]})
    return cases, len(free)


def heldout(args):
    """Honest re-score at the chosen detector setting on held-out val plans (user, chunk 6)."""
    model, cfg = load_detector(args.ckpt)
    choice = yaml.safe_load(CHOICE.read_text(encoding="utf-8"))
    th, mb = choice["threshold"], choice["min_blob"]
    th_rules = R.load_thresholds()
    cases, n_free = build_heldout_cases(args.zip)
    for i, c in enumerate(cases):
        c["probs"] = detector_probs(model, c["plan"], cfg["model"]["K"])
        c["raw"] = R.raw_candidates(c["plan"])
        if (i + 1) % 25 == 0:
            print(f"cached {i + 1}/{len(cases)}", flush=True)
    det = lambda c: proposals(c["probs"], th, mb)
    rules_of = lambda c, keep: R.detect(c["plan"], th_rules, raw=[r for r in c["raw"] if r.rules[0] in keep])
    methods = {"rules only (all rules)": lambda c: R.detect(c["plan"], th_rules, raw=c["raw"]),
               "detector only": det,
               "detector + C8": lambda c: R.merge(det(c) + rules_of(c, ("C8",))),
               "detector + C8/C17/C6/C12": lambda c: R.merge(det(c) + rules_of(c, STRONG_RULES))}
    res = {k: evaluate(cases, f) for k, f in methods.items()}
    final = "detector + C8" if res["detector + C8"][0] <= 0.6 else "detector only"
    n_int = sum(c["mask"] is None for c in cases)
    L = ["", "## Held-out re-score (honest numbers; user, chunk 6)", "",
         f"Setting fixed at threshold {th} / blobs >= {mb} px. Val has only 371 typed plans; {n_free} were NOT used to "
         f"choose the detector setting: {n_int} stay intact, each of the other {n_free - n_int} gets two independent "
         f"omissions of different kinds ({len(cases) - n_int} cases). Caveat: the rule thresholds were tuned on intact "
         "val plans 0-199, which include most of these plans, so the rule columns are slightly optimistic.", "",
         "| method | false alarms / intact plan | mean recall | " + " | ".join(KINDS) + " |",
         "|---|---|---|" + "---|" * len(KINDS)]
    for name, (fa, rec, mean) in res.items():
        cells = [f"{100 * rec[k][0]:.0f}% ({100 * rec[k][1]:.0f}%)" for k in KINDS]
        L.append(f"| {name} | {fa:.2f} | {100 * mean:.0f}% | " + " | ".join(cells) + " |")
    L += ["", f"**Final CHECK setting: {final}** (rule: detector + C8 if <= 0.6 false alarms per plan, else "
          "detector only). The detector already takes the rule violation maps as input, so it is itself the "
          "rules + learned combination."]
    with open(OUT / "detection_eval.md", "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    choice["final"] = final
    choice["extra_rules"] = ["C8"] if final == "detector + C8" else []
    CHOICE.write_text(yaml.safe_dump(choice, sort_keys=False), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    import sys
    if "--heldout" in sys.argv:
        sys.argv.remove("--heldout")
        ap = argparse.ArgumentParser()
        ap.add_argument("--zip", required=True)
        ap.add_argument("--ckpt", required=True)
        heldout(ap.parse_args())
    else:
        main()
