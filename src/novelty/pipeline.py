"""Novelty step 5 - the hybrid pipeline: the shipped Stage 2 pipeline + a path for LARGE omissions.

    result = run(image)                              # automatic
    result = run(image, boxes=[(y0, x0, y1, x1)])    # manual: boxes larger than 100 map px use the new fill

    PARSE  ->  CHECK (shipped rules + detectors)          ->  small flags  ->  shipped small-hole fill (unchanged)
           ->  LOCATE (region locator, whole plan)   \\
           ->  STRUCTURE (missing-wall net, round 2) -+->  region flags ->  new large-hole fill
A region flag swallows the small CHECK flags inside it: a missing room shows up to CHECK as a missing door or
sink, and the region fill redraws that whole area anyway.

Round 1 (configs pipeline_large.yaml = v1, pipeline_large_edit.yaml = v2): region flags from the locator; the
large-hole U-Net draws a sketch, it is noised to t0 and finished by the diffusion model, best of N by rule
violations, then the shipped region restriction (no doors on exterior walls).
Round 2 (a config with a `structure` section): STRUCTURE FIRST.
  * the structure net (src/novelty/structure.py) marks where walls are missing on the whole plan; groups of the
    walls it adds are region flags too (so an omission the locator misses can still be found);
  * in the fill, a wall is drawn wherever the U-Net gives it at least fill.wall_tau probability (not only where
    "wall" beats every other class: a wall whose position is uncertain by a few pixels never wins that vote);
  * the structure net's walls and the U-Net's walls together are the STRUCTURE: they are painted into the sketch,
    the diffusion model then adds the details around them (doors, room types, fixtures), and with fill.keep_walls
    any structure wall the diffusion finish dropped is put back.
Models: src/novelty/models.py, structure.py, locate.py. Settings: the file named by the environment variable
NOVELTY_CONFIG, default configs/novelty/pipeline_large_edit.yaml (v2, the round-1 final).
"""

import os
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from skimage.morphology import skeletonize

from src.novelty.data import ink_map
from src.novelty.locate import load_locator, region_candidates, region_prob
from src.novelty.models import load_large
from src.stage2.complete import large_fill, sampler
from src.stage2.data.classes16 import C16
from src.stage2.pipeline import run as base
from src.utils import PROJECT_ROOT

CFG_FILE = Path(os.environ.get("NOVELTY_CONFIG", PROJECT_ROOT / "configs" / "novelty" / "pipeline_large_edit.yaml"))
BG = C16.index["background"]
WALL = C16.index["wall"]
OPENINGS = [C16.index["door"], C16.index["window"]]


def cfg():
    return yaml.safe_load(CFG_FILE.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def models():
    c = cfg()["checkpoints"]
    out = {"locator": [load_locator(PROJECT_ROOT / p, base.DEVICE) for p in c["locator"] if (PROJECT_ROOT / p).exists()],
           "unet": load_large(PROJECT_ROOT / c["unet"], base.DEVICE),
           "diffusion": load_large(PROJECT_ROOT / c["diffusion"], base.DEVICE)}
    if c.get("structure"):
        from src.novelty.structure import load_structure
        out["structure"] = [load_structure(PROJECT_ROOT / p, base.DEVICE) for p in c["structure"] if (PROJECT_ROOT / p).exists()]
    return out


def near(mask, r):
    return cv2.dilate(mask.astype(np.uint8), np.ones((2 * r + 1,) * 2, np.uint8)) > 0


# ---------------------------------------------------------------- LOCATE

BLOBS = {}                 # box -> blob of the last region_holes() call (the outline prior needs the blob's shape)
STATE = {}                 # per drawing: "image" (for the ink map), "walls" (structure net's added walls), "for" (map shape)


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
    return int(((new == WALL) & hole & ~near(parsed == WALL, 2)).sum())


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


# ---------------------------------------------------------------- STRUCTURE (round 2)

def structure_walls(parsed, ink=None, sc=None, nets=None):
    """The structure net on one parsed map -> (bool map of wall pixels to add, wall heat, area probability)."""
    from src.novelty.structure import decode_walls, predict
    sc = sc or cfg()["structure"]
    heat, area = predict(nets or models()["structure"], parsed, ink, base.DEVICE, tta=sc.get("tta", False))
    return decode_walls(heat, parsed, sc["tau"], sc.get("min_len", 6)), heat, area


def structure_holes(parsed, walls, sc=None):
    """Region flags from the structure net: one box around each group of walls it adds."""
    from src.novelty.structure import wall_clusters
    sc = sc or cfg()["structure"]
    return [{"type": "region", "rules": ["S"], "box": b, "score": 11.0, "point": None, "mask": None}
            for b in wall_clusters(walls, sc.get("join", 8), sc.get("pad", 6), sc.get("min_px", 12))]


def merge_holes(holes):
    """Region flags whose boxes overlap become one flag with the union box (repeated until none overlap)."""
    holes = [dict(h) for h in holes]
    changed = True
    while changed:
        changed = False
        for i in range(len(holes)):
            for j in range(i + 1, len(holes)):
                a, b = holes[i]["box"], holes[j]["box"]
                if a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]:
                    holes[i]["box"] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    holes[i]["rules"] = sorted(set(holes[i]["rules"]) | set(holes[j]["rules"]))
                    holes[i]["score"] = max(holes[i]["score"], holes[j]["score"])
                    holes[i]["mask"] = None
                    del holes[j]
                    changed = True
                    break
            if changed:
                break
    return holes


def all_region_holes(parsed, ink=None):
    """Locator flags + structure flags (if the config has a structure net), overlapping boxes merged.
    Remembers the structure net's walls for the fill (STATE)."""
    c = cfg()
    holes = region_holes(parsed) if c.get("locate", {}).get("use", True) else []
    if c.get("structure") and models().get("structure"):
        walls, _, _ = structure_walls(parsed, ink)
        STATE.update(walls=walls, shape=parsed.shape)
        holes = merge_holes(holes + structure_holes(parsed, walls))
    return holes


# ---------------------------------------------------------------- large-hole FILL

def window_of(hole, nc):
    """(oy, ox, side): the hole's box centred in a square window, side = a multiple of 32, at least nc["window"]."""
    y0, x0, y1, x1 = large_fill.bbox(hole)
    side = max(nc["window"], int(np.ceil((max(y1 - y0, x1 - x0) + 1 + 16) / 32) * 32))
    return (y0 + y1) // 2 - side // 2, (x0 + x1) // 2 - side // 2, side


def outline_walls(sketch, blob, hole, t=3, tol=2):
    """Structure-first prior, round 1 (rejected on tuning drawings): a wall band (t px) just inside the outline of
    the locator's blob, painted into the sketch wherever the sketch has no wall within tol px."""
    k = np.ones((2 * t + 1, 2 * t + 1), np.uint8)
    band = blob & ~(cv2.erode(blob.astype(np.uint8), k) > 0)
    out = sketch.copy()
    out[band & hole & ~near(sketch == WALL, tol)] = WALL
    return out


def thin_walls(mask, min_len=6):
    """A wall mask thinned to its centre line (pieces shorter than min_len px dropped) and redrawn 3 px thick."""
    sk = skeletonize(mask)
    n, comp, st, _ = cv2.connectedComponentsWithStats(sk.astype(np.uint8), connectivity=8)
    keep = np.zeros(mask.shape, bool)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_len:
            keep |= comp == i
    return near(keep, 1)


@torch.no_grad()
def hybrid_fill(label, hole, unet, diffusion=None, t0=700, n=2, steps_from_t0=15, generator=None, nc=None,
                all_samples=False, blob=None, walls=None):
    """label (H, W) map, hole bool mask -> (new map, info). diffusion=None: the U-Net sketch alone.
    all_samples=True also returns every sample pasted into the map (for the best-of-N analysis).
    blob: the locator's blob for this hole (bool map); with it the outline prior is painted into the sketch.
    walls: bool map (plan size) of walls decided elsewhere (the structure net); painted into the sketch.
    nc["wall_tau"]: also draw a wall wherever the U-Net's wall probability is at least this (nc["wall_thin"]:
    thinned to 3 px first). nc["keep_walls"]: structure walls missing after the diffusion finish are put back."""
    nc = nc or cfg()["fill"]
    oy, ox, side = window_of(hole, nc)
    win = large_fill.crop(label, oy, ox, side, BG)
    wh = large_fill.crop(hole.astype(np.uint8), oy, ox, side, 0) > 0
    hidden = large_fill._cond(win, wh).to(base.DEVICE)              # context zeroed in the hole (inpaint models)
    cond = hidden
    if not nc.get("hide_hole", True):                              # edit models: the parsed map stays visible
        full = torch.nn.functional.one_hot(torch.from_numpy(win.astype(np.int64)), C16.n).permute(2, 0, 1).float()
        cond = torch.cat([full[None].to(base.DEVICE), hidden[:, -1:]], 1)
    logits = unet(cond).float()
    sk = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    structure = np.zeros(wh.shape, bool)                           # walls fixed before the diffusion finish
    known = near(win == WALL, 2)
    if nc.get("wall_tau"):
        add = wh & (logits.softmax(1)[0, WALL].cpu().numpy() >= nc["wall_tau"])
        if nc.get("wall_thin"):
            add = (thin_walls(add & ~known) & wh) | (add & known)
        structure |= add
    if walls is not None:
        structure |= (large_fill.crop(walls.astype(np.uint8), oy, ox, side, 0) > 0) & wh
    sk[structure] = WALL
    if blob is not None:
        sk = outline_walls(sk, large_fill.crop(blob.astype(np.uint8), oy, ox, side, 0) > 0, wh)
    sketch = torch.from_numpy(sk.astype(np.int64))[None].to(base.DEVICE)

    def pasted(lab):
        new = label.copy()
        large_fill.paste(new, np.asarray(lab.cpu() if torch.is_tensor(lab) else lab).astype(np.uint8), oy, ox, wh)
        return new

    def kept(lab):                                                  # a sample with the dropped structure walls put back
        out = lab.cpu().numpy().astype(np.uint8)                    # (never over a door / window the diffusion drew:
        if nc.get("keep_walls") and structure.any():                # an opening in a structure wall is a detail it adds)
            out[structure & ~near(out == WALL, 2) & ~np.isin(out, OPENINGS)] = WALL
        return out
    info = {"window": side, "structure_px": int(structure.sum())}
    if diffusion is None:
        return pasted(sketch[0]), info
    c = cond.expand(n, -1, -1, -1)
    samples = sampler.hybrid_sample(diffusion, c, sketch.expand(n, -1, -1), t0=t0, steps=50, generator=generator,
                                    steps_from_t0=steps_from_t0)
    i, counts, agree = sampler.best_of_n(samples, hidden[:, -1:], hidden[:, :C16.n], sketch[0])
    info.update(t0=t0, chosen=i, violations=counts.tolist())
    if all_samples:
        info["samples"] = [pasted(kept(s)) for s in samples]
    return pasted(kept(samples[i])), info


def fill_region(label, hole, generator, fast=None, outline=None):
    """The pipeline's large-hole fill (run_pipeline's large_fill_fn). outline: use the outline prior (default: the
    config's fill.outline_prior)."""
    c = cfg()
    nc, m = c["fill"], models()
    outline = nc.get("outline_prior", False) if outline is None else outline
    t = time.time()
    if window_of(hole, nc)[2] > nc["max_window"]:                 # huge box: the shipped tiled fill
        new, info = large_fill.fill_large(label, hole, "tiled", base.cfg()["fill"], fast, generator)
        return new, dict(info, route="shipped tiled (hole larger than max_window)")
    y0, x0, y1, x1 = large_fill.bbox(hole)
    blob = BLOBS.get((int(y0), int(x0), int(y1), int(x1))) if outline else None
    walls = None
    if c.get("structure") and m.get("structure"):
        if STATE.get("shape") != label.shape or STATE.get("walls") is None:   # manual mode: not computed by a region_fn
            img = STATE.get("image")
            STATE.update(walls=structure_walls(label, None if img is None else ink_map(img, label.shape))[0],
                         shape=label.shape)
        walls = STATE["walls"]
    new, info = hybrid_fill(label, hole, m["unet"], m["diffusion"], nc["t0"], nc["n_samples"], nc["steps_from_t0"],
                            generator, nc, blob=blob, walls=walls)
    info["seconds"] = round(time.time() - t, 1)
    return large_fill.region_restrict(label, new, hole), dict(info, route="novelty large-hole fill")


# ---------------------------------------------------------------- entry point

def run(image, boxes=None, seed=None, fast=True):
    """The hybrid pipeline: run_pipeline with the region flags (locator, structure net) and the new large-hole
    fill switched in."""
    STATE.clear()
    STATE["image"] = image
    region_fn = lambda parsed: all_region_holes(parsed, ink_map(image, parsed.shape))
    return base.run_pipeline(image, boxes=boxes, seed=seed, fast=fast, region_fn=region_fn, large_fill_fn=fill_region)
