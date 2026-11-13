"""Novelty step 5 - the hybrid pipeline: the shipped Stage 2 pipeline + a path for LARGE omissions.

    result = run(image)                              # automatic
    result = run(image, boxes=[(y0, x0, y1, x1)])    # manual: boxes larger than 100 map px use the new fill

    PARSE  ->  CHECK (shipped rules + detectors)        ->  small flags  ->  shipped small-hole fill (unchanged)
           ->  LOCATE (region locator, whole plan)      ->  region flags ->  new large-hole fill
A region flag swallows the small CHECK flags inside it: a missing room shows up to CHECK as a missing door or
sink, and the region fill redraws that whole area anyway.

Large-hole fill (fill_region): the hole's box centred in a window of >= 256 px (most of the plan), the new U-Net
draws a sketch, it is noised to t0 and finished by the new diffusion model, best-of-N by rule violations, then
the shipped region restriction (no doors on exterior walls). Models: src/novelty/models.py, trained on the
parser's maps of erased drawings (src/novelty/train_fill.py). Settings: configs/novelty/pipeline_large.yaml.
"""

import os
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from src.novelty.locate import load_locator, region_candidates, region_prob
from src.novelty.models import load_large
from src.stage2.complete import large_fill, sampler
from src.stage2.data.classes16 import C16
from src.stage2.pipeline import run as base
from src.utils import PROJECT_ROOT

CFG_FILE = Path(os.environ.get("NOVELTY_CONFIG", PROJECT_ROOT / "configs" / "novelty" / "pipeline_large.yaml"))
BG = C16.index["background"]


def cfg():
    return yaml.safe_load(CFG_FILE.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def models():
    c = cfg()["checkpoints"]
    return {"locator": [load_locator(PROJECT_ROOT / p, base.DEVICE) for p in c["locator"] if (PROJECT_ROOT / p).exists()],
            "unet": load_large(PROJECT_ROOT / c["unet"], base.DEVICE),
            "diffusion": load_large(PROJECT_ROOT / c["diffusion"], base.DEVICE)}


# ---------------------------------------------------------------- LOCATE

BLOBS = {}                 # box -> blob of the last region_holes() call (the outline prior needs the blob's shape)


def box_mask(shape, box):
    m = np.zeros(shape, bool)
    m[box[0]:box[2] + 1, box[1]:box[3] + 1] = True
    return m


def sketch_change(parsed, box, unet=None, nc=None):
    """The "would the fill change anything?" check of a region flag: hide the box, let the large-hole U-Net draw
    it, and count the wall pixels it draws that are NOT within 2 px of a wall already in the parsed map.
    A complete area is redrawn as it was (few new wall pixels); a missing room gets its walls (many)."""
    hole = box_mask(parsed.shape, box)
    if window_of(hole, nc or cfg()["fill"])[2] > (nc or cfg()["fill"])["max_window"]:
        return 0
    new, _ = hybrid_fill(parsed, hole, unet or models()["unet"], None, nc=nc)
    wall = C16.index["wall"]
    near = cv2.dilate((parsed == wall).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return int(((new == wall) & hole & ~near).sum())


def region_holes(parsed, prob=None, lc=None):
    """Region flags of the locator as pipeline holes (box = the hole handed to the large-hole fill).
    locate.min_change > 0 keeps only flags whose sketch_change is at least that many wall pixels."""
    lc = lc or cfg()["locate"]
    if prob is None:
        prob = region_prob(models()["locator"], parsed, base.DEVICE)
    holes = [{"type": "region", "rules": ["L"], "box": c.box, "score": c.score, "point": None, "mask": c.mask}
             for c in region_candidates(prob, lc["threshold"], lc["min_area"], lc["box_pad"])]
    BLOBS.clear()
    BLOBS.update({tuple(int(v) for v in h["box"]): h["mask"] for h in holes})
    if lc.get("min_change", 0) > 0:
        for h in holes:
            h["change"] = sketch_change(parsed, h["box"])
        holes = [h for h in holes if h["change"] >= lc["min_change"]]
    return holes


# ---------------------------------------------------------------- large-hole FILL

def window_of(hole, nc):
    """(oy, ox, side): the hole's box centred in a square window, side = a multiple of 32, at least nc["window"]."""
    y0, x0, y1, x1 = large_fill.bbox(hole)
    side = max(nc["window"], int(np.ceil((max(y1 - y0, x1 - x0) + 1 + 16) / 32) * 32))
    return (y0 + y1) // 2 - side // 2, (x0 + x1) // 2 - side // 2, side


def outline_walls(sketch, blob, hole, t=3, tol=2):
    """Structure-first prior: a wall band (t px) just inside the outline of the locator's blob, painted into the
    sketch wherever the sketch has no wall within tol px. For a missing room the blob is the room, so its outline
    is where the room's walls were (tuning drawings: walls on the blob outline alone bring back 57% of a missing
    room's wall pixels within 2 px)."""
    wall = C16.index["wall"]
    k = np.ones((2 * t + 1, 2 * t + 1), np.uint8)
    band = blob & ~(cv2.erode(blob.astype(np.uint8), k) > 0)
    near = cv2.dilate((sketch == wall).astype(np.uint8), np.ones((2 * tol + 1,) * 2, np.uint8)) > 0
    out = sketch.copy()
    out[band & hole & ~near] = wall
    return out


@torch.no_grad()
def hybrid_fill(label, hole, unet, diffusion=None, t0=700, n=2, steps_from_t0=15, generator=None, nc=None,
                all_samples=False, blob=None):
    """label (H, W) map, hole bool mask -> (new map, info). diffusion=None: the U-Net sketch alone.
    all_samples=True also returns every sample pasted into the map (for the best-of-N analysis).
    blob: the locator's blob for this hole (bool map); with it the outline prior is painted into the sketch."""
    nc = nc or cfg()["fill"]
    oy, ox, side = window_of(hole, nc)
    win = large_fill.crop(label, oy, ox, side, BG)
    wh = large_fill.crop(hole.astype(np.uint8), oy, ox, side, 0) > 0
    hidden = large_fill._cond(win, wh).to(base.DEVICE)              # context zeroed in the hole (inpaint models)
    cond = hidden
    if not nc.get("hide_hole", True):                              # edit models: the parsed map stays visible
        full = torch.nn.functional.one_hot(torch.from_numpy(win.astype(np.int64)), C16.n).permute(2, 0, 1).float()
        cond = torch.cat([full[None].to(base.DEVICE), hidden[:, -1:]], 1)
    sketch = unet.predict(cond)
    if blob is not None:
        wb = large_fill.crop(blob.astype(np.uint8), oy, ox, side, 0) > 0
        sk = outline_walls(sketch[0].cpu().numpy().astype(np.uint8), wb, wh)
        sketch = torch.from_numpy(sk.astype(np.int64))[None].to(base.DEVICE)

    def pasted(lab):
        new = label.copy()
        large_fill.paste(new, lab.cpu().numpy().astype(np.uint8), oy, ox, wh)
        return new
    if diffusion is None:
        return pasted(sketch[0]), {"window": side}
    c = cond.expand(n, -1, -1, -1)
    samples = sampler.hybrid_sample(diffusion, c, sketch.expand(n, -1, -1), t0=t0, steps=50, generator=generator,
                                    steps_from_t0=steps_from_t0)
    i, counts, agree = sampler.best_of_n(samples, hidden[:, -1:], hidden[:, :C16.n], sketch[0])
    info = {"window": side, "t0": t0, "chosen": i, "violations": counts.tolist()}
    if all_samples:
        info["samples"] = [pasted(s) for s in samples]
    return pasted(samples[i]), info


def fill_region(label, hole, generator, fast=None):
    """The pipeline's large-hole fill (run_pipeline's large_fill_fn)."""
    nc, m = cfg()["fill"], models()
    t = time.time()
    if window_of(hole, nc)[2] > nc["max_window"]:                 # huge box: the shipped tiled fill
        new, info = large_fill.fill_large(label, hole, "tiled", base.cfg()["fill"], fast, generator)
        return new, dict(info, route="shipped tiled (hole larger than max_window)")
    blob = BLOBS.get(tuple(int(v) for v in large_fill.bbox(hole))) if nc.get("outline_prior") else None
    new, info = hybrid_fill(label, hole, m["unet"], m["diffusion"], nc["t0"], nc["n_samples"], nc["steps_from_t0"],
                            generator, nc, blob=blob)
    info["seconds"] = round(time.time() - t, 1)
    return large_fill.region_restrict(label, new, hole), dict(info, route="novelty large-hole fill")


# ---------------------------------------------------------------- entry point

def run(image, boxes=None, seed=None, fast=True):
    """The hybrid pipeline: run_pipeline with the locator and the new large-hole fill switched in."""
    return base.run_pipeline(image, boxes=boxes, seed=seed, fast=fast, region_fn=region_holes, large_fill_fn=fill_region)
