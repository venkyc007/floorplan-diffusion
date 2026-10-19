"""Tune the whole CHECK on PARSED TUNING drawings (parser from configs/stage2/pipeline.yaml, erasure v2).

    python -m src.stage2.detect.check_tune --zip parser_val.zip --tuning tuning_ids.txt [--intact 40] [--omitted 80]
        [--budget 0.6] [--det-min 0.25] [--write]
        -> outputs/stage2/check_tune.md (+ --write: configs/stage2/detector_threshold.yaml)

Drawings: typed TUNING val drawings (file order): the first `intact` stay intact (false alarms), the next
`omitted` get two separate omissions each, kinds in turn over image_eval.KINDS (seed 11), made as in image_eval.
Every drawing is parsed once; all settings are then scored from stored proposals (collect / boxes / score, also
used by check_curve.py):
  detector  threshold x min blob (THS x BLOBS; off = no proposals)
  C8        minimum room area (C8_AREAS px; inf = off)
  R5        the configured gap rule, on / off
  door ink  threshold (INK_THS; 0 = off) + per-drawing guard on all doors; candidates at ALL doors or only at the
            relevant ones (R5's pairs: door_ink.relevant; user, 2026-10-19)
Found = a proposal box overlaps the omission (resized to the parse, grown 3 px).
Choice (user, 2026-10-19): budget 0.6 false alarms per intact drawing in total. Stage 1: rules + ink (detector off)
with the best mean recall within budget - det_min (so >= det_min stays for the detector). Stage 2: with those rules
fixed, the detector threshold / blob with the best mean recall of the whole within the budget.
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
from src.stage2.detect.door_ink import door_ink_scores, guarded, relevant
from src.stage2.detect.gaps import closed_pairs, gap_candidates, instance_areas
from src.stage2.detect.image_eval import KINDS, omit
from src.stage2.pipeline.run import cfg, models, parse
from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "stage2"
CHOICE = PROJECT_ROOT / "configs" / "stage2" / "detector_threshold.yaml"
THS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
BLOBS = [20, 40, 80, 160]
C8_AREAS = [0, 200, 400, 700, 1000, 1500, 2756, np.inf]
INK_THS = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
OFF = None                                     # detector off


def read(z, name, flag):
    return cv2.imdecode(np.frombuffer(z.read(name), np.uint8), flag)


def collect(img, choice):
    """Everything needed to score any setting on one drawing."""
    label, _, _ = parse(img, cfg()["parse"])
    m = models()
    probs = detector_probs(m["detector"], label, m["detector_K"])
    det = {(t, b): [p.box for p in proposals(probs, t, b)] for t in THS for b in BLOBS}
    c8 = [(c.box, c.score) for c in R.discrete_candidates(label) if c.rules[0] == "C8"]
    g = choice.get("gap_rule", {})
    r5 = [c.box for c in gap_candidates(label, open_rate=g.get("open_rate", 0.10), need_area=g.get("need_area"),
                                        small_other=g.get("small_other"), min_len=g.get("min_len"))]
    bad, area = closed_pairs(open_rate=g.get("open_rate", 0.10)), instance_areas(label)
    ink = [(R.box_of(d, label.shape), s, relevant(label, d, g, bad, area)) for d, s in door_ink_scores(img, label)]
    return label.shape, det, c8, r5, ink


def boxes(rec, S):
    """Proposal boxes of one drawing under setting S = {det: (t, b) or None, c8, r5, ink, rel}."""
    _, det, c8, r5, ink = rec
    out = list(det.get(S["det"], [])) if S["det"] is not OFF else []
    out += [bx for bx, s in c8 if s >= S["c8"]]
    out += r5 if S["r5"] else []
    if S["ink"] > 0 and not guarded([s for _, s, _ in ink], S["ink"]):      # guard on all doors of the drawing
        out += [bx for bx, s, rel in ink if s < S["ink"] and (rel or not S["rel"])]
    return out


def hit(bxs, rm):
    return any(rm[y0:y1 + 1, x0:x1 + 1].any() for y0, x0, y1, x1 in bxs)


def score(intact, omitted, S):
    """-> dict(S, fa=false alarms per intact drawing, mean=mean recall over kinds, <kind>=recall)."""
    fa = float(np.mean([len(R.merge([R.Candidate("x", bx) for bx in boxes(r, S)])) for r in intact]))
    per = {kd: [] for kd in KINDS}
    for kd, rm, r in omitted:
        per[kd].append(hit(boxes(r, S), rm))
    rec = {kd: float(np.mean(v)) for kd, v in per.items() if v}
    return {**S, "fa": fa, "mean": float(np.mean(list(rec.values()))), **rec}


def fmt(r):
    det = "off" if r["det"] is OFF else f"{r['det'][0]} / {r['det'][1]} px"
    c8 = "off" if r["c8"] == np.inf else str(int(r["c8"]))
    ink = "off" if r["ink"] == 0 else f"{r['ink']} ({'relevant doors' if r['rel'] else 'all doors'})"
    return (f"| {det} | {c8} | {'on' if r['r5'] else 'off'} | {ink} | {r['fa']:.2f} | {100 * r['mean']:.0f}% | " +
            " | ".join(f"{100 * r[kd]:.0f}%" if kd in r else "-" for kd in KINDS) + " |")


HDR = ["| detector (threshold / min blob) | C8 min area | R5 | ink check | false alarms / intact | mean recall | " +
       " | ".join(KINDS) + " |", "|---|---|---|---|---|---|" + "---|" * len(KINDS)]


def build(zip_path, ids, n_intact, n_omitted, seed, choice):
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
                    kind = KINDS[(k + step) % len(KINDS)]
                    o = omit(kind, img, lbl, rng)
                    if o is not None:
                        rec = collect(o[0], choice)
                        rm = cv2.resize(o[1].astype(np.uint8), (rec[0][1], rec[0][0]), interpolation=cv2.INTER_NEAREST) > 0
                        omitted.append((kind, grown(rm, 3), rec))
                        break
        print(f"intact {len(intact)}, omissions {len(omitted)} ({time.time() - t0:.0f} s)", flush=True)
    return intact, omitted


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--tuning", required=True)
    ap.add_argument("--intact", type=int, default=40)
    ap.add_argument("--omitted", type=int, default=80)
    ap.add_argument("--budget", type=float, default=0.6)
    ap.add_argument("--det-min", type=float, default=0.25)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    choice = yaml.safe_load(CHOICE.read_text(encoding="utf-8"))
    intact, omitted = build(args.zip, args.tuning, args.intact, args.omitted, 11, choice)
    # stage 1: rules + ink, detector off, within budget - det_min
    rules = [score(intact, omitted, {"det": OFF, "c8": a, "r5": r5, "ink": k, "rel": rel})
             for a, r5, k, rel in itertools.product(C8_AREAS, (True, False), INK_THS, (True, False))
             if not (k == 0 and rel)]
    ok1 = [r for r in rules if r["fa"] <= args.budget - args.det_min]
    best1 = max(ok1, key=lambda r: (r["mean"], -r["fa"]))
    # stage 2: detector inside what is left
    base = {k: best1[k] for k in ("c8", "r5", "ink", "rel")}
    full = [score(intact, omitted, {**base, "det": (t, b)}) for t in THS for b in BLOBS]
    ok2 = [r for r in full if r["fa"] <= args.budget]
    best = max(ok2 + [best1], key=lambda r: (r["mean"], -r["fa"]))
    # ink check false alarms before / after the restriction (ink alone, at the chosen threshold, guard on)
    k = best1["ink"] or 1.5
    ink_all = score(intact, omitted, {"det": OFF, "c8": np.inf, "r5": False, "ink": k, "rel": False})
    ink_rel = score(intact, omitted, {"det": OFF, "c8": np.inf, "r5": False, "ink": k, "rel": True})
    det_only = [score(intact, omitted, {"det": (t, b), "c8": np.inf, "r5": False, "ink": 0, "rel": False})
                for t in THS for b in BLOBS]
    L = ["# CHECK tuned on parsed TUNING drawings (parser v2B, detector v2, erasure v2)", "",
         f"{len(intact)} intact + {len(omitted)} omissions on tuning drawings. Budget {args.budget} false alarms per intact "
         f"drawing in total; rules + ink chosen first within {args.budget - args.det_min:.2f}, then the detector inside "
         "what is left (user, 2026-10-19). Found = a proposal overlaps the omission.", "", "## Chosen", ""] + HDR + [fmt(best)]
    L += ["", f"Stage 1 (rules + ink, detector off; budget {args.budget - args.det_min:.2f}), top 8:", ""] + HDR + \
         [fmt(r) for r in sorted(ok1, key=lambda r: (-r["mean"], r["fa"]))[:8]]
    L += ["", "Stage 2 (detector added to the chosen rules; budget "
          f"{args.budget}), top 8:", ""] + HDR + [fmt(r) for r in sorted(ok2, key=lambda r: (-r["mean"], r["fa"]))[:8]]
    L += ["", f"## Door-ink check before / after the restriction (ink alone, threshold {k}, guard on)", ""] + HDR + \
         [fmt(ink_all), fmt(ink_rel)]
    L += ["", "## Detector v2 alone (best within 0.15 / 0.25 / 0.35 false alarms)", ""] + HDR
    for cap in (0.15, 0.25, 0.35):
        c = [r for r in det_only if r["fa"] <= cap]
        if c:
            L.append(fmt(max(c, key=lambda r: (r["mean"], -r["fa"]))))
    (OUT / "check_tune.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    if args.write:
        choice.update(threshold=float(best["det"][0]) if best["det"] is not OFF else 1.01,
                      min_blob=int(best["det"][1]) if best["det"] is not OFF else 20,
                      c8_min_area=None if best["c8"] == np.inf else float(best["c8"]),
                      ink_threshold=float(best["ink"]), ink_relevant_only=bool(best["rel"]),
                      extra_rules=[s for s, on in (("C8", best["c8"] != np.inf), ("R5", best["r5"]), ("INK", best["ink"] > 0)) if on],
                      final=f"detector v2 + C8 + R5 + ink check; budget {args.budget} (rules + ink first, detector after)",
                      check_tuning=f"outputs/stage2/check_tune.md (parsed tuning drawings; budget {args.budget}, "
                                   f"detector >= {args.det_min}; user 2026-10-19)")
        CHOICE.write_text(yaml.safe_dump(choice, sort_keys=False), encoding="utf-8")
        print("written", CHOICE)


if __name__ == "__main__":
    main()
