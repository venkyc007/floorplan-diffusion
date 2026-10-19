"""Stage 2 end-to-end: a black-and-white floor plan DRAWING in, missing components found and drawn.

    result = run_pipeline(image)                       # automatic: parse -> detect -> fill -> verify
    result = run_pipeline(image, boxes=[(y0, x0, y1, x1)])   # manual: fill the given boxes (image px)

image: (H, W) greyscale uint8 (0 = ink, 255 = paper), any resolution and line width.
Steps
  PARSE   our parser (ResNet-34 U-Net) at scale 1; the predicted walls' thickness t is measured and,
          if t is not ~8 px, the drawing is parsed again at the configured scale closest to 8 / t
          (same choice as trying every scale, fewer passes). The label map is then rescaled so walls
          are 3 px (completion scale).
  CHECK   the learned detector (its input already includes the rule violation maps) at the chosen
          threshold / blob size, plus C8 "room without a door" if the final setting says so
          (configs/stage2/detector_threshold.yaml) -> holes with a type.
  FILL    per hole, a 128 x 128 window: 16-class U-Net sketch -> re-noise to t0 (400 / 600 / 700 by hole
          size, as in Stage 1) -> K1 constrained diffusion with x0 guidance -> best-of-N verifier.
  VERIFY  the verifier's violation counts are returned per hole.
Returns a dict: completed (completion-scale label map), parsed (same scale, before filling), holes
(with type, reason, box, t0, violations), additions (class -> mask), overlay (RGB drawing with the
additions in red; doors drawn as gap + leaf + arc), messages (text list), timings (s per step).
"""

import time
from functools import lru_cache

import cv2
import numpy as np
import torch
import yaml

from src.stage2.complete import sampler
from src.stage2.complete.models16 import build16
from src.stage2.complete.sketch_prior import SIZES, apply_prior
from src.stage2.constraints.rules import components
from src.stage2.data.classes16 import C16
from src.stage2.data.plan_stats import door_side_pair
from src.stage2.detect import rules as R
from src.stage2.detect.detection_eval import detector_probs, load_detector, proposals
from src.stage2.detect.door_ink import ink_candidates
from src.stage2.detect.gaps import gap_candidates
from src.stage2.parse.train import Parser, predict_safe
from src.utils import PROJECT_ROOT

CFG_FILE = PROJECT_ROOT / "configs" / "stage2" / "pipeline.yaml"
I = C16.index


def cfg():
    with open(CFG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- models (loaded once)

@lru_cache(maxsize=None)
def models():
    c = cfg()["checkpoints"]
    load = lambda p: torch.load(PROJECT_ROOT / p, map_location="cpu", weights_only=False)
    ck = load(c["parser"])
    parser = Parser(ck["config"]["model"], encoder_weights=None)
    parser.load_state_dict(ck["model"])
    detector, dcfg = load_detector(PROJECT_ROOT / c["detector"])
    out = {"parser": parser.eval(), "detector": detector, "detector_K": dcfg["model"]["K"]}
    for name in ("unet", "diffusion"):
        ck = load(c[name])
        m = build16(ck["config"]["model"])
        m.load_state_dict(ck["model"])
        out[name] = m.eval()
    return out


# ---------------------------------------------------------------- PARSE

def wall_thickness(label):
    """Median wall thickness (px): twice the distance-transform value on the wall's ridge pixels."""
    wall = (label == I["wall"]).astype(np.uint8)
    if wall.sum() < 20:
        return float("nan")
    dt = cv2.distanceTransform(wall, cv2.DIST_L2, 3)
    ridge = (dt > 0) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-6)
    return float(2 * np.median(dt[ridge]))


def parse_at(image, scale, max_side):
    h, w = image.shape
    s = min(scale, max_side / max(h, w))
    img = cv2.resize(image, (max(32, int(round(w * s))), max(32, int(round(h * s)))),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    H, W = -(-img.shape[0] // 32) * 32, -(-img.shape[1] // 32) * 32
    pad = np.full((H, W), 255, np.uint8)
    pad[:img.shape[0], :img.shape[1]] = img
    x = torch.from_numpy(pad)[None, None].float() / 255
    with torch.no_grad():
        lab = predict_safe(models()["parser"], x, amp=False, tile=1024, overlap=64)[:img.shape[0], :img.shape[1]]
    return lab.astype(np.uint8), s


def parse(image, pc):
    lab, s = parse_at(image, 1.0, pc["max_side"])
    t = wall_thickness(lab)
    if np.isfinite(t) and abs(t - pc["wall_px"]) > 1.5:
        want = pc["wall_px"] / t
        best = min(pc["scales"], key=lambda x: abs(np.log(x / want)))
        if best != 1.0:
            lab, s = parse_at(image, best, pc["max_side"])
            t = wall_thickness(lab)
    f = pc["completion_wall_px"] / t if np.isfinite(t) and t > 0 else 3 / 8
    comp = cv2.resize(lab, (max(8, int(round(lab.shape[1] * f))), max(8, int(round(lab.shape[0] * f)))),
                      interpolation=cv2.INTER_NEAREST)
    return comp, s * f, {"parse_scale": s, "wall_px_at_parse_scale": t}


# ---------------------------------------------------------------- CHECK

def detect_parts(label, image=None):
    """CHECK proposals by source: {"detector": [...], "C8": [...], "R5": [...], "INK": [...]} (rules only if
    configured; the door-ink check needs the drawing)."""
    choice = yaml.safe_load((PROJECT_ROOT / cfg()["detect"]["settings"]).read_text(encoding="utf-8"))
    m = models()
    probs = detector_probs(m["detector"], label, m["detector_K"])
    parts = {"detector": proposals(probs, choice["threshold"], choice["min_blob"])}
    extra = choice.get("extra_rules", [])
    if "C8" in extra:                                    # room cut off: per building part
        raw = [c for c in R.discrete_candidates(label) if c.rules[0] == "C8"]
        th = R.load_thresholds()
        if choice.get("c8_min_area") is not None:        # re-tuned on parsed drawings (user, 2026-10-19)
            th = dict(th, C8=choice["c8_min_area"])
        parts["C8"] = R.detect(label, th, raw=raw)
    if "R5" in extra:                                    # doorless gap between rooms that need a door
        g = choice.get("gap_rule", {})
        parts["R5"] = gap_candidates(label, open_rate=g.get("open_rate", 0.10), need_area=g.get("need_area"),
                                     small_other=g.get("small_other"), min_len=g.get("min_len"))
    if "INK" in extra and image is not None:             # door drawn as an opening only (no leaf / arc)
        parts["INK"] = ink_candidates(image, label, choice["ink_threshold"],
                                      gap=choice.get("gap_rule", {}) if choice.get("ink_relevant_only") else None)
    return parts


def detect_holes(label, image=None):
    return R.merge([p for v in detect_parts(label, image).values() for p in v])


# ---------------------------------------------------------------- FILL

def window_around(label, box, win):
    """128 x 128 window (background-padded) containing the box; returns window, origin, hole mask."""
    y0, x0, y1, x1 = box
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    oy, ox = cy - win // 2, cx - win // 2
    h, w = label.shape
    out = np.full((win, win), I["background"], np.uint8)
    sy0, sx0, sy1, sx1 = max(0, oy), max(0, ox), min(h, oy + win), min(w, ox + win)
    out[sy0 - oy:sy1 - oy, sx0 - ox:sx1 - ox] = label[sy0:sy1, sx0:sx1]
    hole = np.zeros((win, win), np.uint8)
    hole[max(0, y0 - oy):min(win, y1 - oy + 1), max(0, x0 - ox):min(win, x1 - ox + 1)] = 1
    return out, (oy, ox), hole


def fill_hole(label, box, fc, generator, fast=None, typ=None, rules=()):
    m = models()
    win, origin, hole = window_around(label, box, fc["window"])
    t = torch.from_numpy(win.astype(np.int64))[None]
    target = torch.nn.functional.one_hot(t, C16.n).permute(0, 3, 1, 2).float()
    mask = torch.from_numpy(hole).float()[None, None]
    cond = torch.cat([target * (1 - mask), mask], 1)
    building = max(1, int(np.isin(label, C16.group("BOUNDARY") + C16.group("S") + C16.group("F")).sum()))
    share = hole.sum() / building
    t0 = fc["t0"][0] if share < fc["size_bins"][0] else (fc["t0"][1] if share < fc["size_bins"][1] else fc["t0"][2])
    t0 = fc.get("t0_fixed") or t0
    with torch.no_grad():
        sketch = m["unet"].predict(cond)
    if fc.get("sketch_prior") and typ:                   # type-guided fill: the missing component painted
        sk = apply_prior(sketch[0].numpy().astype(np.uint8), hole > 0, typ, rules)   # into the sketch
        sketch = torch.from_numpy(sk.astype(np.int64))[None]
    kw = {"n": fc["n_samples"]} if fast is None else {"n": fast["n_samples"], "steps_from_t0": fast["steps_from_t0"],
                                                         "guide_last": fast["guide_last"]}
    out, info = sampler.fill(m["diffusion"], cond, sketch=sketch, t0=t0, eta_guidance=fc["eta_guidance"],
                             steps=fc["ddim_steps"], generator=generator, **kw)
    new = label.copy()
    oy, ox = origin
    h, w = label.shape
    res = out[0].numpy().astype(np.uint8)
    ys, xs = np.nonzero(hole)
    gy, gx = ys + oy, xs + ox
    ok = (gy >= 0) & (gy < h) & (gx >= 0) & (gx < w)
    new[gy[ok], gx[ok]] = res[ys[ok], xs[ok]]
    return new, {"t0": t0, "size_share": round(float(share), 4), **info[0]}


# ---------------------------------------------------------------- outputs

REAL = ["wall", "door", "window", "railing"] + list(C16.cfg["groups"]["F"])
FIXTURES = list(C16.cfg["groups"]["F"])
ROOM_ORDER = ["bedroom", "bathroom", "kitchen", "other_room", "living", "hallway"]   # "the room that had no door"


def host_room(label, m):
    """Most common room type under / around a mask (on the map before the addition), or None."""
    ring = cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    v = label[ring & np.isin(label, C16.group("S"))]
    return C16.names[int(np.bincount(v).argmax())] if len(v) else None


def clip_door(door, length=None):
    """An added door clipped to ONE wall line at the median door length (user, 2026-10-19): the band of rows
    (or columns) holding its main run along its longer side, and +- length / 2 around that run's centre. An
    L-shaped door spread over two walls keeps only the longer arm."""
    length = length or SIZES["door"][1]
    ys, xs = np.nonzero(door)
    horizontal = (xs.max() - xs.min()) >= (ys.max() - ys.min())
    across, along = (ys, xs) if horizontal else (xs, ys)
    counts = np.bincount(across - across.min())
    band = counts >= 0.5 * counts.max()
    sel = band[across - across.min()]
    c = int(np.median(along[sel]))
    sel &= np.abs(along - c) <= length // 2
    out = np.zeros_like(door)
    out[ys[sel], xs[sel]] = True
    return out


def restrict(before, after, typ):
    """Type-restricted fill (user, chunk 8): inside a detected hole keep ONLY the detected type and
    revert every other changed pixel to the map before the fill.
      door / window / railing / a fixture  -> its largest new component only (one door per hole;
                                              a door clipped to one wall line at median length)
      wall                                 -> new wall pixels
      room                                 -> new walls, doors and room types (its walls, type and door)
      manual                               -> everything the model drew
    A fixture is dropped if the room it lands in is not compatible with it (C15 table)."""
    if typ == "manual":
        return after
    changed = after != before
    if typ == "room":
        keep = changed & np.isin(after, [I["wall"], I["door"]] + C16.group("S"))
    elif typ == "wall":
        keep = changed & (after == I["wall"])
    else:
        comps = components(changed & (after == I[typ]))
        keep = max(comps, key=lambda m: m.sum()) if comps else np.zeros(after.shape, bool)
        if typ == "door" and keep.any():
            keep = clip_door(keep)
        if typ in FIXTURES and keep.any():
            room = host_room(before, keep)
            if room not in R.CONST.get("fixture_rooms", {}).get(typ, []):
                keep[:] = False
    return np.where(keep, after, before)


def describe_additions(before, after, holes):
    """One line per real addition (wall, door, window, railing, fixture), never space classes and no
    duplicates; for a missing room, the room type that was created is named."""
    msgs = []
    k = 5
    for h in holes:
        y0, x0, y1, x1 = h["box"]
        sub = np.zeros(after.shape, bool)
        sub[y0:y1 + 1, x0:x1 + 1] = True
        changed = sub & (after != before)
        rules = set(h.get("rules") or [])
        lines = []
        for d in components(changed & (after == I["door"])):
            names = [C16.names[p] for p in (door_side_pair(after, d, k) or [])]
            rooms = sorted({n for n in names if n in ROOM_ORDER}, key=ROOM_ORDER.index)
            where = " / ".join(names) if names else "the"
            if "C8" in rules and rooms:
                lines.append(f"{rooms[0]} had no door -> door added on the {where} wall")
            elif "R5" in rules and len(names) == 2:
                lines.append(f"open doorway between {names[0]} and {names[1]} had no door -> door added")
            else:
                lines.append(f"door added on the {where} wall")
        for c in REAL:
            if c == "door":
                continue
            for m in components(changed & (after == I[c])):
                room = host_room(before, m) or "the plan"
                where = {"wall": f"next to {room}", "window": f"on the {room} wall"}.get(c, f"in {room}")
                lines.append(f"{c} added {where}")
        if h["type"] == "room":
            created = [C16.names[c] for c in np.unique(after[changed]) if c in C16.group("S")]
            if created:
                lines.append(f"missing room -> room drawn as {', '.join(created)}")
        if not lines:
            what = f"{h['type']} missing ({'+'.join(sorted(rules))})" if rules else "manual box"
            lines.append(f"{what} -> nothing added (the model kept the drawing as it was)")
        msgs += [m for m in lines if m not in msgs]
    return msgs


def overlay(image, before, after, scale):
    """Additions in red on the original drawing; doors drawn as gap + leaf + quarter arc."""
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
    H, W = image.shape
    added = (after != before) & ~np.isin(after, C16.group("S") + C16.group("OUT"))
    up = lambda m: cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    non_door = up(added & (after != I["door"]))
    rgb[non_door] = 0.3 * rgb[non_door] + 0.7 * np.array([220, 0, 0])
    for d in components(added & (after == I["door"])):
        ys, xs = np.nonzero(up(d))
        if len(ys) == 0:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        rgb[y0:y1 + 1, x0:x1 + 1] = 255                               # the gap in the wall
        L = int(max(y1 - y0, x1 - x0))
        horizontal = (x1 - x0) >= (y1 - y0)
        img = rgb.astype(np.uint8).copy()
        th = max(1, int(round(1.5 / scale)))
        if horizontal:
            cv2.line(img, (int(x0), int(y1)), (int(x0), int(y1 + L)), (220, 0, 0), th)
            cv2.ellipse(img, (int(x0), int(y1)), (L, L), 0, 0, 90, (220, 0, 0), th)
        else:
            cv2.line(img, (int(x1), int(y0)), (int(x1 + L), int(y0)), (220, 0, 0), th)
            cv2.ellipse(img, (int(x1), int(y0)), (L, L), 0, 0, 90, (220, 0, 0), th)
        rgb = img.astype(np.float32)
    return rgb.astype(np.uint8)


# ---------------------------------------------------------------- entry point

def run_pipeline(image, boxes=None, seed=None, fast=False):
    """fast=True: the live-app settings (configs/stage2/pipeline.yaml `fast`)."""
    c = cfg()
    times = {}
    t = time.time()
    parsed, scale, pinfo = parse(image, c["parse"])
    times["parse"] = time.time() - t
    t = time.time()
    if boxes is None:
        holes = [{"type": p.type, "rules": p.rules, "box": p.box, "score": p.score} for p in detect_holes(parsed, image)]
    else:
        holes = [{"type": "manual", "rules": [], "box": tuple(int(round(v * scale)) for v in b)} for b in boxes]
    mh = c["fill"]["max_hole"]
    for h in holes:                                              # keep holes window-sized
        y0, x0, y1, x1 = h["box"]
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        h["box"] = (max(cy - mh // 2, y0), max(cx - mh // 2, x0), min(cy + mh // 2, y1), min(cx + mh // 2, x1))
    times["detect"] = time.time() - t
    t = time.time()
    g = torch.Generator().manual_seed(c["fill"]["seed"] if seed is None else seed)
    completed = parsed.copy()
    for h in holes:
        new, info = fill_hole(completed, h["box"], c["fill"], g, c["fast"] if fast else None, h["type"], h["rules"])
        kept = restrict(completed, new, h["type"])
        y0, x0, y1, x1 = h["box"]                                # debug record: map before, raw hybrid output and
        sl = (slice(max(0, y0 - 16), y1 + 17), slice(max(0, x0 - 16), x1 + 17))   # what the restriction kept
        h["raw"] = {"origin": (sl[0].start, sl[1].start), "before": completed[sl].copy(), "hybrid": new[sl].copy(),
                    "kept": kept[sl].copy()}
        completed = kept
        h.update(info)
    times["fill"] = time.time() - t
    additions = {C16.names[k]: (completed != parsed) & (completed == k) for k in np.unique(completed[completed != parsed])}
    return {"parsed": parsed, "completed": completed, "holes": holes, "additions": additions,
            "overlay": overlay(image, parsed, completed, scale), "messages": describe_additions(parsed, completed, holes),
            "scale": scale, "parse_info": pinfo, "timings": {k: round(v, 1) for k, v in times.items()}}
