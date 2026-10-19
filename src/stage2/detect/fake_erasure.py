"""Control test: does detector v3 react to the ERASURE TRACE rather than to a missing component? (2026-10-19)

    python -m src.stage2.detect.fake_erasure --zip parser_val.zip --heldout heldout_ids.txt [--n 40]
        -> outputs/stage2/fake_erasure.md

Intact held-out drawings; on each, two fake erasures of things that are NOT components (nothing is missing after):
  word   one room-name word (text_erase.text_mask group) erased exactly as text is erased (inpainting)
  furniture  a furniture drawing (bed, table; not a component) erased exactly as a fixture is (erasure.clear_zone,
         box + 40%, median + noise fill): the same cleaned-patch trace a fixture omission leaves
  blank  a blank 24 x 24 px patch of room floor filled the same way (a weak control: little changes)
Each version is parsed; detectors v2 and v3 run at the chosen thresholds (detector_threshold.yaml). Firing rate =
share of spots with a proposal box overlapping the spot (resized to the parse, grown 3 px). Baseline = the same
spots on the untouched drawing (chance overlap). Firing above baseline = reaction to the trace.
"""

import argparse
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.stage2.data.classes16 import C16
from src.stage2.data.erasure import LONG, clear_zone, erase, grown, symbol_zone
from src.stage2.detect.detection_eval import detector_probs, proposals
from src.stage2.detect.detector_v3 import v3_probs
from src.stage2.parse.text_erase import text_mask
from src.stage2.pipeline.run import cfg, models, parse
from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "stage2"
CHOICE = PROJECT_ROOT / "configs" / "stage2" / "detector_threshold.yaml"


def read(z, name, flag):
    return cv2.imdecode(np.frombuffer(z.read(name), np.uint8), flag)


def word_spot(img, lbl, rng):
    m = text_mask(img, lbl)
    n, comp = cv2.connectedComponents(grown(m, 6).astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    k = int(rng.integers(1, n))
    return m & (comp == k)


def blank_spot(img, lbl, rng, size=24, margin=12):
    room = np.isin(lbl, C16.group("S"))
    ok = cv2.erode(room.astype(np.uint8), np.ones((2 * margin + size,) * 2, np.uint8)) > 0
    ok &= cv2.erode((img > 200).astype(np.uint8), np.ones((size + 4,) * 2, np.uint8)) > 0   # blank paper / fill
    pts = np.argwhere(ok)
    if not len(pts):
        return None
    cy, cx = pts[rng.integers(len(pts))]
    m = np.zeros(img.shape, bool)
    m[cy - size // 2:cy + size // 2, cx - size // 2:cx + size // 2] = True
    return m


def furniture_spot(img, lbl, rng, ink=170, lo=40, hi=200, inside=0.95):
    """A furniture drawing (bed, table, sofa): a dark connected piece, 40-200 px across, >= 95% on room labels and
    not touching any labelled component (it is not a component, so nothing is missing once it is erased)."""
    n, comp, st, _ = cv2.connectedComponentsWithStats((img < ink).astype(np.uint8), connectivity=8)
    side = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
    ok = (side >= lo) & (side <= hi)
    ok[0] = False
    solid = grown(np.isin(lbl, C16.group("BOUNDARY") + C16.group("F")), 2)
    ok[np.unique(comp[solid])] = False
    in_room = np.bincount(comp[np.isin(lbl, C16.group("S"))].ravel(), minlength=n)
    ok &= in_room >= inside * st[:, cv2.CC_STAT_AREA]
    ids = np.nonzero(ok)[0]
    return (comp == ids[rng.integers(len(ids))]) if len(ids) else None


def fires(props, spot, shape):
    rm = cv2.resize(spot.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    rm = grown(rm, 3)
    return any(rm[p.box[0]:p.box[2] + 1, p.box[1]:p.box[3] + 1].any() for p in props)


def detect(img, choice):
    label, _, _ = parse(img, cfg()["parse"])
    m = models()
    p2 = proposals(detector_probs(m["detector"], label, m["detector_K"]), choice["threshold"], choice["min_blob"])
    p3 = proposals(v3_probs(m["detector_v3"], img, label), choice["threshold_v3"], choice["min_blob_v3"])
    return label.shape, p2, p3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    choice = yaml.safe_load(CHOICE.read_text(encoding="utf-8"))
    keep = set(Path(args.heldout).read_text(encoding="utf-8").split())
    z = zipfile.ZipFile(args.zip)
    names = sorted(n[:-8] for n in z.namelist() if n.endswith("_img.png") and n[:-8] in keep)
    rng = np.random.default_rng(3)
    res = {k: {"v2": [], "v3": [], "v2 base": [], "v3 base": []} for k in ("word", "furniture", "blank")}
    done = 0
    for n in names:
        if done >= args.n:
            break
        lbl = read(z, f"{n}_lbl.png", cv2.IMREAD_UNCHANGED)
        if (lbl == 255).any():
            continue
        img = read(z, f"{n}_img.png", cv2.IMREAD_GRAYSCALE)
        shape, b2, b3 = detect(img, choice)
        for kind in ("word", "furniture", "blank"):
            spot = {"word": word_spot, "furniture": furniture_spot, "blank": blank_spot}[kind](img, lbl, rng)
            if spot is None or not spot.any():
                continue
            if kind == "word":
                fake = erase(img, spot, lbl, wall_px=8)
            else:                                        # exactly as a fixture erasure (box + 40%, median + noise)
                fake = clear_zone(img, spot, symbol_zone(spot, img.shape), lbl, LONG["symbol"], median=True)
            sh, p2, p3 = detect(fake, choice)
            r = res[kind]
            r["v2"].append(fires(p2, spot, sh))
            r["v3"].append(fires(p3, spot, sh))
            r["v2 base"].append(fires(b2, spot, shape))
            r["v3 base"].append(fires(b3, spot, shape))
        done += 1
        print(f"{done}/{args.n} {n}", flush=True)
    L = ["# Control: do the detectors fire on erasure TRACES? (fake erasures, intact held-out drawings)", "",
         f"{done} intact held-out drawings. Fake erasures of non-components (nothing is missing afterwards): a room-name "
         "word (inpainted as text erasure), a furniture drawing and a blank floor patch (both erased as a fixture is). Firing = a "
         "detector proposal overlaps the spot; baseline = the same spot on the untouched drawing. Real omissions "
         "(held-out, same thresholds): detectors found 58% (detection_eval.md).", "",
         "| fake erasure | spots | v3 fires | v3 baseline | v2 fires | v2 baseline |", "|---|---|---|---|---|---|"]
    for k, r in res.items():
        if r["v3"]:
            L.append(f"| {k} | {len(r['v3'])} | {100 * np.mean(r['v3']):.0f}% | {100 * np.mean(r['v3 base']):.0f}% | "
                     f"{100 * np.mean(r['v2']):.0f}% | {100 * np.mean(r['v2 base']):.0f}% |")
    (OUT / "fake_erasure.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
