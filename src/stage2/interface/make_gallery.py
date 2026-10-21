"""Build the Stage 2 example gallery for the app: held-out VAL drawings with one component erased.

    python -m src.stage2.interface.make_gallery        -> data/app/stage2_gallery/ (git-ignored)

Which drawings: the held-out val plans (configs/stage2/val_split/heldout_ids.txt, never used to choose a setting),
in the same fixed random order as the end-to-end validation (seed 20), untyped plans skipped. For each
[kind, count] of configs/stage2/app.yaml `gallery_plan`, the next drawings that have such a component are taken
in that order (no picking by hand). The erasure is the validation's own generator (src/stage2/detect/image_eval.omit),
so the app shows exactly the kind of case the numbers in outputs/stage2/ were measured on.

Saved per case i:  <i>_drawing.png  (the erased drawing, greyscale, as the app's input)
                   <i>_original.png (the drawing before the erasure, to compare)
                   <i>_labels.png   (ground-truth 16-class label map at the drawing's scale)
                   <i>_removed.png  (the erased components, 255 = removed; empty for intact drawings), so the app
                                    can score a run with the validation's own definitions (e2e_val.score)
and index.json: id, kind, expected type, the omission's box in drawing px (y0, x0, y1, x1; null if intact)
and its metadata (location, position, size bin; src/stage2/data/omission_meta.py).
"""

import json
import zipfile

import cv2
import numpy as np
import yaml

from src.stage2.detect.image_eval import omit
from src.stage2.pipeline.e2e_val import CHECK_SET, HELDOUT, read
from src.utils import PROJECT_ROOT

CFG = yaml.safe_load((PROJECT_ROOT / "configs" / "stage2" / "app.yaml").read_text(encoding="utf-8"))
ORDER_SEED = 20          # the end-to-end validation's drawing order (src/stage2/pipeline/e2e_val.py pick)
BOX_PAD = 4              # px around the omission (as in the validation's region box)


def heldout_order():
    """Held-out val ids in the validation's fixed random order, check-set drawings left out."""
    held = HELDOUT.read_text(encoding="utf-8").split()
    order = [held[i] for i in np.random.default_rng(ORDER_SEED).permutation(len(held))]
    return [p for p in order if p not in CHECK_SET]


def box_of(mask, shape, pad=BOX_PAD):
    ys, xs = np.nonzero(mask)
    return [int(max(0, ys.min() - pad)), int(max(0, xs.min() - pad)),
            int(min(shape[0] - 1, ys.max() + pad)), int(min(shape[1] - 1, xs.max() + pad))]


def build(z, plan, seed):
    """[(case dict, erased drawing, original drawing, labels, removed mask)] following `plan` ([kind, count])."""
    rng = np.random.default_rng(seed)
    order, used, cases = heldout_order(), set(), []
    for kind, count in plan:
        got = 0
        for pid in order:
            if got >= count:
                break
            if pid in used:
                continue
            lbl = read(z, f"{pid}_lbl.png", cv2.IMREAD_UNCHANGED)
            if (lbl == 255).any():                       # untyped plan (no class for some pixels)
                continue
            img = read(z, f"{pid}_img.png", cv2.IMREAD_GRAYSCALE)
            if kind == "intact":
                erased, box, typ, meta = img, None, None, {"type": "intact"}
                removed = np.zeros(lbl.shape, bool)
            else:
                o = omit(kind, img, lbl, rng)
                if o is None:                            # this drawing has no such component
                    continue
                erased, removed, typ, meta, area = o[0], o[1], o[2], o[4], o[5]
                box = box_of(area, lbl.shape)
            used.add(pid)
            got += 1
            case = {"id": len(cases), "plan_id": pid, "kind": kind, "expected_type": typ, "box": box,
                    "meta": {k: meta.get(k, "") for k in ("type", "location", "position", "size_bin", "size_pct")}}
            cases.append((case, erased, img, lbl, removed))
    return cases


def main():
    out = PROJECT_ROOT / CFG["gallery_dir"]
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PROJECT_ROOT / CFG["val_zip"]) as z:
        cases = build(z, CFG["gallery_plan"], CFG["gallery_seed"])
    for case, erased, original, lbl, removed in cases:
        i = case["id"]
        cv2.imwrite(str(out / f"{i:02d}_drawing.png"), erased)
        cv2.imwrite(str(out / f"{i:02d}_original.png"), original)
        cv2.imwrite(str(out / f"{i:02d}_labels.png"), lbl)
        cv2.imwrite(str(out / f"{i:02d}_removed.png"), removed.astype(np.uint8) * 255)
        print(f"{i:2d} {case['plan_id']:34s} {case['kind']:13s} {case['meta'].get('size_bin', '')}")
    (out / "index.json").write_text(json.dumps([c for c, *_ in cases], indent=1), encoding="utf-8")
    print(f"{len(cases)} cases -> {out}")


if __name__ == "__main__":
    main()
