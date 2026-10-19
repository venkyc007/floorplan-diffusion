"""Held-out trade-off: recall vs false alarms per intact drawing for three CHECK families (user, 2026-10-19).

    python -m src.stage2.detect.check_curve --completion completion_val.zip --drawings parser_val.zip
        -> outputs/stage2/check_curve.png, outputs/stage2/check_curve.md

Cases: exactly image_eval's held-out drawings (59 intact + 100 drawings x 2 omissions, seed 5, erasure v2), each
parsed once; proposals stored with check_tune.collect and scored at many settings (check_tune.score).
Families (6 operating points each):
  (i)   rules + ink: C8 >= 2756 px; C8 at the chosen area; + R5; + ink at 0.5 / the chosen threshold / 3.0
  (ii)  detector v2 only, at the chosen min blob, thresholds 0.98 / 0.95 / 0.9 / 0.8 / 0.7 / 0.5
  (iii) everything: the chosen rules + ink + detector at those thresholds
The chosen setting (configs/stage2/detector_threshold.yaml) is marked.
"""

import argparse
import time
import zipfile

import cv2
import numpy as np
import yaml

from src.stage2.data.erasure import grown
from src.stage2.detect.check_tune import CHOICE, HDR, OFF, OUT, collect, fmt, read, score
from src.stage2.detect.detection_eval import build_heldout_cases
from src.stage2.detect.image_eval import KINDS, omit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--completion", required=True)
    ap.add_argument("--drawings", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    choice = yaml.safe_load(CHOICE.read_text(encoding="utf-8"))
    held, _ = build_heldout_cases(args.completion)
    ids_intact = [c["pid"] for c in held if c["mask"] is None]
    ids_omit = list(dict.fromkeys(c["pid"] for c in held if c["mask"] is not None))
    del held
    if args.limit:
        ids_intact, ids_omit = ids_intact[:args.limit], ids_omit[:args.limit]
    z = zipfile.ZipFile(args.drawings)
    rng = np.random.default_rng(5)                       # same sequence as image_eval
    intact, omitted, t0 = [], [], time.time()
    for pid in ids_intact:
        intact.append(collect(read(z, f"{pid}_img.png", cv2.IMREAD_GRAYSCALE), choice))
    for j, pid in enumerate(ids_omit):
        img = read(z, f"{pid}_img.png", cv2.IMREAD_GRAYSCALE)
        lbl = read(z, f"{pid}_lbl.png", cv2.IMREAD_UNCHANGED)
        for k in (2 * j, 2 * j + 1):
            for step in range(len(KINDS)):
                kind = KINDS[(k + step) % len(KINDS)]
                o = omit(kind, img, lbl, rng)
                if o is not None:
                    rec = collect(o[0], choice)
                    rm = cv2.resize(o[1].astype(np.uint8), (rec[0][1], rec[0][0]), interpolation=cv2.INTER_NEAREST) > 0
                    omitted.append((kind, grown(rm, 3), rec))
                    break
        print(f"omitted {j + 1}/{len(ids_omit)} ({time.time() - t0:.0f} s)", flush=True)
    on = choice.get("extra_rules", [])
    c8 = choice.get("c8_min_area") or np.inf if "C8" in on else np.inf
    k = choice.get("ink_threshold", 0) if "INK" in on else 0
    k2 = choice.get("ink_threshold_interior", 0) if "INK" in on else 0
    r6 = (choice.get("r6_scale") or np.inf) if "R6" in on else np.inf
    r7 = (choice.get("r7_min_width") or np.inf) if "R7" in on else np.inf
    blob = choice.get("min_blob", 40)
    det = (choice["threshold"], blob) if choice.get("threshold", 1.01) <= 1 else OFF
    chosen_rules = {"c8": c8, "r5": "R5" in on, "ink": k, "ink2": k2, "r6": r6, "r7": r7}
    none = {"det": OFF, "c8": np.inf, "r5": False, "ink": 0, "ink2": 0, "r6": np.inf, "r7": np.inf}
    ths = [0.98, 0.95, 0.9, 0.8, 0.7, 0.5]
    fam = {        # (i): the rules switched on one after another, ending at the chosen rule set
        "(i) rules + ink": [{**none, "c8": 2756}, {**none, "c8": c8}, {**none, "c8": c8, "r5": True},
                            {**none, "c8": c8, "r5": True, "ink": k or 1.5},
                            {**none, "c8": c8, "r5": True, "ink": k or 1.5, "ink2": k2},
                            {**chosen_rules, "det": OFF}],
        "(ii) detector v2 only": [{**none, "det": (t, blob)} for t in ths],
        "(iii) everything": [{**chosen_rules, "det": (t, blob)} for t in ths],
    }
    res = {f: [score(intact, omitted, S) for S in Ss] for f, Ss in fam.items()}
    chosen = score(intact, omitted, {**chosen_rules, "det": det})
    L = ["# Held-out trade-off: recall vs false alarms (parser v2B, detector v2, erasure v2)", "",
         f"{len(intact)} intact + {len(omitted)} omissions (image_eval's held-out cases). Chosen setting: see "
         "'Chosen point' at the end. Figure: outputs/stage2/check_curve.png.", ""]
    for f, rs in res.items():
        L += [f"## {f}", ""] + HDR + [fmt(r) for r in rs] + [""]
    L += ["## Chosen point (per type)", ""] + HDR + [fmt(chosen)]
    (OUT / "check_curve.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for (f, rs), m in zip(res.items(), ("o-", "s-", "^-")):
        ax.plot([r["fa"] for r in rs], [100 * r["mean"] for r in rs], m, label=f)
    ax.plot([chosen["fa"]], [100 * chosen["mean"]], "r*", ms=16, label="chosen")
    ax.axvline(0.6, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("false alarms per intact drawing (held-out)")
    ax.set_ylabel("mean recall over omission types (%)")
    ax.set_title("CHECK on held-out drawings: recall vs false alarms")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "check_curve.png", dpi=100)


if __name__ == "__main__":
    main()
