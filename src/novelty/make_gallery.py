"""Build the example gallery of the app's novelty tab: held-out VAL drawings with a large or medium part erased.

    python -m src.novelty.make_gallery        -> data/app/novelty_gallery/ (git-ignored; CubiCasa5K is not redistributed)

The cases are EXACTLY the held-out evaluation's cases (src/novelty/eval_e2e.py): the same drawing order
(drawing_ids, seed 30), the same omission generator (omit_in_bin) and the same random seed per drawing ([31, index],
with the small, medium and large omissions drawn in the same sequence). So the app shows cases whose results are in
outputs/novelty/e2e_heldout_round4.csv, and the index stores those results next to each case. Drawings are taken in
that order for each [size, count] of configs/novelty/app.yaml `gallery_plan`; nothing is picked by hand.

Saved per case i (same layout as the Stage 2 gallery, src/stage2/interface/make_gallery.py):
  <i>_drawing.png (erased drawing, the app's input), <i>_original.png, <i>_labels.png (true 16-class map),
  <i>_removed.png (erased components, 255), <i>_area.png (the erased area, for the recovered / sound scores)
and index.json: id, plan id, size, kind, expected type, box (drawing px), metadata, held-out results per version.
"""

import csv
import json
import zipfile

import cv2
import numpy as np
import yaml

from src.novelty.eval_e2e import BINS, drawing_ids
from src.stage2.detect.omission_mix import omit_in_bin
from src.stage2.interface.make_gallery import box_of
from src.utils import PROJECT_ROOT

CFG = yaml.safe_load((PROJECT_ROOT / "configs" / "novelty" / "app.yaml").read_text(encoding="utf-8"))


def heldout_results():
    """{(plan id, target): {version label: row}} from the saved held-out evaluation (empty if not there)."""
    p = PROJECT_ROOT / CFG["heldout_results"]
    if not p.exists():
        return {}
    arm_of = {"hybrid " + v.split(" ")[0] + " auto": v for v in CFG["versions"]}   # "hybrid v4c auto" -> label
    out = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["arm"] in arm_of:
            keep = {k: r.get(k, "") for k in ("found", "fixed", "recovered", "s_sound", "n_region_flags", "added")}
            out.setdefault((r["pid"], r["target"]), {})[arm_of[r["arm"]]] = keep
    return out


def build(z):
    """[(case dict, erased, original, labels, removed, area)] following the gallery plan."""
    need = {size: n for size, n in CFG["gallery_plan"]}
    got = {size: 0 for size in need}
    results, cases = heldout_results(), []
    read = lambda n, f: cv2.imdecode(np.frombuffer(z.read(n), np.uint8), f)
    for idx, pid in enumerate(drawing_ids("heldout", z)):
        if all(got[s] >= need[s] for s in need):
            break
        lbl = read(f"{pid}_lbl.png", cv2.IMREAD_UNCHANGED)
        if (lbl == 255).any():                            # untyped plan: skipped by the evaluation too
            continue
        img = read(f"{pid}_img.png", cv2.IMREAD_GRAYSCALE)
        rng = np.random.default_rng([31, idx])            # as eval_e2e.run_cases: one rng, bins drawn in order
        drawn = {b: omit_in_bin(b, img, lbl, rng) for b in BINS}
        size = next((s for s in ("intact", "large", "medium") if s in need and got[s] < need[s]
                     and (s == "intact" or drawn[s] is not None)), None)
        if size is None:
            continue
        got[size] += 1
        if size == "intact":
            erased, removed, area, typ, meta, box = img, np.zeros(lbl.shape, bool), np.zeros(lbl.shape, bool), None, \
                {"type": "intact"}, None
        else:
            o = drawn[size]
            erased, removed, typ, meta, area = o[0], o[1], o[2], o[4], o[5]
            box = box_of(area | removed, lbl.shape)
        case = {"id": len(cases), "plan_id": pid, "size": size, "kind": meta["type"], "expected_type": typ,
                "box": box, "meta": {k: meta.get(k, "") for k in ("type", "location", "size_bin", "size_pct")},
                "heldout": results.get((pid, size), {})}
        cases.append((case, erased, img, lbl, removed, area))
    return cases


def main():
    out = PROJECT_ROOT / CFG["gallery_dir"]
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PROJECT_ROOT / CFG["val_zip"]) as z:
        cases = build(z)
    for case, erased, original, lbl, removed, area in cases:
        i = case["id"]
        cv2.imwrite(str(out / f"{i:02d}_drawing.png"), erased)
        cv2.imwrite(str(out / f"{i:02d}_original.png"), original)
        cv2.imwrite(str(out / f"{i:02d}_labels.png"), lbl)
        cv2.imwrite(str(out / f"{i:02d}_removed.png"), removed.astype(np.uint8) * 255)
        cv2.imwrite(str(out / f"{i:02d}_area.png"), area.astype(np.uint8) * 255)
        print(f"{i:2d} {case['plan_id']:34s} {case['size']:7s} {case['kind']:13s} held-out: {case['heldout']}")
    (out / "index.json").write_text(json.dumps([c for c, *_ in cases], indent=1), encoding="utf-8")
    print(f"{len(cases)} cases -> {out}")


if __name__ == "__main__":
    main()
