"""Re-tune the whole CHECK on PARSED TUNING drawings (user, 2026-10-19): parser from configs/stage2/pipeline.yaml
(v2B), erasure v2, budget ~0.4 false alarms per intact drawing for detector + C8 + R5 + door-ink together.

    python -m src.stage2.detect.check_tune --zip parser_val.zip --tuning tuning_ids.txt [--intact 40] [--omitted 80] [--write]
        -> outputs/stage2/check_tune.md (+ --write: configs/stage2/detector_threshold.yaml)

Drawings: typed TUNING val drawings (file order): the first `intact` stay intact (false alarms), the next
`omitted` get two separate omissions each, kinds in turn over image_eval.KINDS (seed 11), made as in image_eval
(erasure v2). Every drawing is parsed once; then all settings are scored from stored proposals:
  detector  threshold x min blob (THS x BLOBS; threshold 1.01 = off)
  C8        minimum room area (C8_AREAS px; inf = off)
  R5        the configured gap rule, on / off
  door ink  threshold on the ink score (INK_THS; 0 = off), with the per-drawing guard (door_ink.guarded)
Found = a proposal box overlaps the omission (resized to the parse, grown 3 px). Choice = the highest mean recall
over the omission kinds with <= budget false alarms per intact drawing (ties: fewer false alarms).
"""

import argparse
import itertools
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.stage2.data.erasure import grown
from src.stage2.detect import rules as R
from src.stage2.detect.detection_eval import detector_probs, proposals
from src.stage2.detect.door_ink import door_ink_scores, guarded
from src.stage2.detect.gaps import gap_candidates
from src.stage2.detect.image_eval import KINDS, omit
from src.stage2.pipeline.run import cfg, models, parse
from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "stage2"
CHOICE = PROJECT_ROOT / "configs" / "stage2" / "detector_threshold.yaml"
THS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 1.01]
BLOBS = [20, 40, 80, 160]
C8_AREAS = [0, 200, 400, 700, 1000, 1500, 2756, np.inf]
INK_THS = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]


def read(z, name, flag):
    return cv2.imdecode(np.frombuffer(z.read(name), np.uint8), flag)


def collect(img, choice):
    """Everything needed to score any setting on one drawing."""
    label, _, _ = parse(img, cfg()["parse"])
    m = models()
    probs = detector_probs(m["detector"], label, m["detector_K"])
    det = {(t, b): [p.box for p in proposals(probs, t, b)] for t in THS[:-1] for b in BLOBS}
    c8 = [(c.box, c.score) for c in R.discrete_candidates(label) if c.rules[0] == "C8"]
    g = choice.get("gap_rule", {})
    r5 = [c.box for c in gap_candidates(label, open_rate=g.get("open_rate", 0.10), need_area=g.get("need_area"),
                                        small_other=g.get("small_other"), min_len=g.get("min_len"))]
    ink = [(R.box_of(d, label.shape), s) for d, s in door_ink_scores(img, label)]
    return label.shape, det, c8, r5, ink


def boxes(rec, t, b, a, r5_on, k):
    _, det, c8, r5, ink = rec
    out = list(det.get((t, b), []))
    out += [bx for bx, s in c8 if s >= a]
    out += r5 if r5_on else []
    if not guarded([s for _, s in ink], k):              # per-drawing ink guard (user, 2026-10-19)
        out += [bx for bx, s in ink if s < k]
    return out


def hit(bxs, rm):
    return any(rm[y0:y1 + 1, x0:x1 + 1].any() for y0, x0, y1, x1 in bxs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--tuning", required=True)
    ap.add_argument("--intact", type=int, default=40)
    ap.add_argument("--omitted", type=int, default=80)
    ap.add_argument("--budget", type=float, default=0.4)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    choice = yaml.safe_load(CHOICE.read_text(encoding="utf-8"))
    keep = set(Path(args.tuning).read_text(encoding="utf-8").split())
    z = zipfile.ZipFile(args.zip)
    names = sorted(n[:-8] for n in z.namelist() if n.endswith("_img.png") and n[:-8] in keep)
    rng = np.random.default_rng(11)
    intact, omitted, t0 = [], [], time.time()
    for n in names:
        if len(intact) >= args.intact and len(omitted) >= 2 * args.omitted:
            break
        lbl = read(z, f"{n}_lbl.png", cv2.IMREAD_UNCHANGED)
        if (lbl == 255).any():
            continue
        img = read(z, f"{n}_img.png", cv2.IMREAD_GRAYSCALE)
        if len(intact) < args.intact:
            intact.append(collect(img, choice))
        else:
            j = len(omitted) // 2
            for k in (2 * j, 2 * j + 1):
                for step in range(len(KINDS)):
                    kind = KINDS[(k + step) % len(KINDS)]
                    o = omit(kind, img, lbl, rng)
                    if o is not None:
                        rec = collect(o[0], choice)
                        rm = cv2.resize(o[1].astype(np.uint8), (rec[0][1], rec[0][0]), interpolation=cv2.INTER_NEAREST) > 0
                        omitted.append((kind, grown(rm, 3), rec))
                        break
        print(f"intact {len(intact)}, omissions {len(omitted)} ({time.time() - t0:.0f} s)", flush=True)
    results = []
    for (t, b), a, r5_on, k in itertools.product([(t, b) for t in THS for b in BLOBS], C8_AREAS, (True, False), INK_THS):
        if t == 1.01 and b != BLOBS[0]:
            continue
        fa = np.mean([len(R.merge([R.Candidate("x", bx) for bx in boxes(r, t, b, a, r5_on, k)])) for r in intact])
        per = {kd: [] for kd in KINDS}
        for kd, rm, r in omitted:
            per[kd].append(hit(boxes(r, t, b, a, r5_on, k), rm))
        rec = {kd: float(np.mean(v)) for kd, v in per.items() if v}
        results.append({"det": (t, b), "c8": a, "r5": r5_on, "ink": k, "fa": float(fa), "mean": float(np.mean(list(rec.values()))),
                        **rec})
    ok = [r for r in results if r["fa"] <= args.budget]
    best = max(ok, key=lambda r: (r["mean"], -r["fa"])) if ok else min(results, key=lambda r: r["fa"])
    def fmt(r):
        t, b = r["det"]
        det = "off" if t > 1 else f"{t} / {b} px"
        c8 = "off" if r["c8"] == np.inf else str(int(r["c8"]))
        ink = "off" if r["ink"] == 0 else str(r["ink"])
        return (f"| {det} | {c8} | {'on' if r['r5'] else 'off'} | {ink} | {r['fa']:.2f} | {100 * r['mean']:.0f}% | " +
                " | ".join(f"{100 * r[kd]:.0f}%" if kd in r else "-" for kd in KINDS) + " |")
    hdr = ["| detector (threshold / min blob) | C8 min area | R5 | ink threshold | false alarms / intact | mean recall | " +
           " | ".join(KINDS) + " |", "|---|---|---|---|---|---|" + "---|" * len(KINDS)]
    old = next((r for r in results if r["det"] == (choice["threshold"], choice["min_blob"]) and r["c8"] == 2756
                and r["r5"] and r["ink"] == 0), None)
    single = lambda **kw: max([r for r in ok if all(r[k_] == v for k_, v in kw.items())] or [{"mean": 0, "fa": 0}],
                              key=lambda r: (r["mean"], -r["fa"]))
    L = ["# CHECK re-tuned on parsed TUNING drawings (parser v2B, erasure v2)", "",
         f"{len(intact)} intact + {len(omitted)} omissions on tuning drawings; budget {args.budget} false alarms per intact "
         "drawing for the whole CHECK. Found = a proposal overlaps the omission.", "", "## Chosen", ""] + hdr + [fmt(best)]
    if old:
        L += ["", "Previous setting (0.9 / 40 px, C8 >= 2756 px, R5, no ink check) on the same drawings:", ""] + hdr + [fmt(old)]
    L += ["", "## Best within budget, top 10", ""] + hdr + [fmt(r) for r in sorted(ok, key=lambda r: (-r["mean"], r["fa"]))[:10]]
    L += ["", "## What each source adds (best within budget with only that source on)", ""] + hdr
    for name, kw in (("detector only", {"c8": np.inf, "r5": False, "ink": 0}),
                     ("C8 only", {"det": (1.01, BLOBS[0]), "r5": False, "ink": 0}),
                     ("ink only", {"det": (1.01, BLOBS[0]), "c8": np.inf, "r5": False})):
        r = single(**kw)
        if "det" in r:
            L.append(fmt(r).replace("| ", f"| ({name}) ", 1))
    (OUT / "check_tune.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    if args.write:
        choice.update(threshold=float(best["det"][0]), min_blob=int(best["det"][1]),
                      c8_min_area=None if best["c8"] == np.inf else float(best["c8"]),
                      ink_threshold=float(best["ink"]),
                      extra_rules=[s for s, on in (("C8", best["c8"] != np.inf), ("R5", best["r5"]), ("INK", best["ink"] > 0)) if on],
                      check_tuning="outputs/stage2/check_tune.md (parsed tuning drawings, parser v2B, erasure v2, "
                                   f"budget {args.budget} for the whole CHECK; user 2026-10-19)")
        CHOICE.write_text(yaml.safe_dump(choice, sort_keys=False), encoding="utf-8")
        print("written", CHOICE)


if __name__ == "__main__":
    main()
