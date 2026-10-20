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
  GAP     a door flag (R5 / door-ink / detector) on a break in a wall line no door fits (shorter than Lmin_door
          or longer than Lmax_door; src/stage2/detect/flag_gap.py) becomes "wall missing": the gap is closed.
  EXISTS  a flag whose type (door / window / railing / fixture) is already in the parse at the flag: the parser
          read the symbol, the drawing lacks it -> reported "<type> symbol missing in the drawing -> added" and
          drawn in red; no fill (user, 2026-10-20).
  FILL    per hole, a 128 x 128 window: 16-class U-Net sketch (+ the missing component painted in at the flag
          point) -> re-noise to t0 -> K1 constrained diffusion -> best-of-N verifier -> type restriction.
  VERIFY  the verifier's violation counts are returned per hole.
Returns a dict: completed (completion-scale label map), parsed (same scale, before filling), holes
(with type, rules, box, point, t0, violations; "existing" / "gap_len" when those steps applied), additions
(class -> mask), marked (bool: parsed components reported as added, EXISTS), overlay (RGB drawing with the
additions and marked components in red; doors drawn as gap + leaf + arc), messages (text list), timings.
"""

import time
from functools import lru_cache

import cv2
import numpy as np
import torch
import yaml

from src.stage2.complete import sampler
from src.stage2.complete.large_fill import fill_large
from src.stage2.complete.models16 import build16
from src.stage2.complete.sketch_prior import SIZES, apply_prior
from src.stage2.constraints.rules import components
from src.stage2.data.classes16 import C16
from src.stage2.data.plan_stats import door_side_pair
from src.stage2.detect import rules as R
from src.stage2.detect.detection_eval import detector_probs, load_detector, proposals
from src.stage2.detect.detector_v3 import load_v3, v3_probs
from src.stage2.detect.door_ink import ink_candidates
from src.stage2.detect.flag_gap import flagged_gap, gap_verdict
from src.stage2.detect.gaps import gap_candidates
from src.stage2.detect.merged_rooms import merged_room_candidates
from src.stage2.detect.narrow_gaps import gap_candidates_narrow
from src.stage2.detect.symbol_ink import wall_ink_candidates, window_ink_candidates
from src.stage2.detect.window_runs import window_candidates
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
    if c.get("detector_v3") and (PROJECT_ROOT / c["detector_v3"]).exists():
        out["detector_v3"] = load_v3(PROJECT_ROOT / c["detector_v3"])
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

def detect_parts(label, image=None, choice=None):
    """CHECK proposals by source: {"detector", "C8", "R5", "INK", "R6", "R7": [...]} (rules only if configured;
    the door-ink check needs the drawing). `choice`: settings dict (default: the configured file)."""
    choice = choice or yaml.safe_load((PROJECT_ROOT / cfg()["detect"]["settings"]).read_text(encoding="utf-8"))
    m = models()
    which = choice.get("detector_model", "v2")          # v2, v3 or both (tuned, user 2026-10-19)
    parts = {"detector": []}
    if which in ("v2", "both"):
        probs = detector_probs(m["detector"], label, m["detector_K"])
        parts["detector"] += proposals(probs, choice["threshold"], choice["min_blob"])
    if which in ("v3", "both") and image is not None and "detector_v3" in m:
        probs = v3_probs(m["detector_v3"], image, label)
        parts["detector"] += proposals(probs, choice["threshold_v3"], choice["min_blob_v3"])
    # railings: 0% recall in every held-out run -> manual mode only (user, 2026-10-19)
    parts["detector"] = [p for p in parts["detector"] if p.type not in choice.get("manual_only_types", [])]
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
                                      gap=choice.get("gap_rule", {}) if choice.get("ink_relevant_only") else None,
                                      threshold_interior=choice.get("ink_threshold_interior", 0.0))
    if "R6" in extra:                                    # windowless exterior span -> window missing
        wr = yaml.safe_load((PROJECT_ROOT / "configs" / "stage2" / "window_rule.yaml").read_text(encoding="utf-8"))
        parts["R6"] = window_candidates(label, wr["thresholds"], wr["need_area"], SIZES["window"][1],
                                        choice.get("r6_scale", 1.0))
    if "R7" in extra:                                    # wall break narrower than any door -> wall missing
        parts["R7"] = gap_candidates_narrow(label, min_width=choice.get("r7_min_width", 1))
    if "WINK" in extra and image is not None:            # parsed window without a window symbol in the drawing
        parts["WINK"] = window_ink_candidates(image, label, choice["window_ink_dmax"], choice["window_gap_ink"])
    if "WALLINK" in extra and image is not None:         # parsed wall with no ink under it
        parts["WALLINK"] = wall_ink_candidates(image, label, choice["wall_ink_min_len"])
    if "R8" in extra:                                    # oversized / merged room -> region missing
        parts["R8"] = merged_room_candidates(label, choice.get("r8_area_mult") or np.inf, bool(choice.get("r8_mix")))
    return parts


def detect_holes(label, image=None):
    choice = yaml.safe_load((PROJECT_ROOT / cfg()["detect"]["settings"]).read_text(encoding="utf-8"))
    if choice.get("use_judge") and image is not None:     # generous flagger + U-Net judge (judge.py)
        from src.stage2.detect.judge import judged_flags
        return judged_flags(label, image, choice)
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


def flag_point(p):
    """Where a flag points (y, x): the detector's heat peak, else the box centre; None for C8-only flags
    (a whole room without a door: no location)."""
    if set(p.rules) == {"C8"}:
        return None
    if p.peak is not None:
        return tuple(p.peak)
    y0, x0, y1, x1 = p.box
    return ((y0 + y1) // 2, (x0 + x1) // 2)


GAP_SOURCES = {"R5", "INK", "detector"}


def gap_rule(label, h):
    """Door flag on a wall break that no door fits -> wall hole on that gap (user, 2026-10-20)."""
    if h["type"] != "door" or not GAP_SOURCES & set(h["rules"]):
        return
    g = flagged_gap(label, h["box"], h["point"])
    if g is None:
        return
    h["gap_len"] = g["length"]
    if gap_verdict(g["length"]) == "wall":
        h["type"], h["gap"], h["box"] = "wall", g["mask"], R.box_of(g["mask"], label.shape, pad=2)


MARKABLE = ["door", "window", "railing"] + list(C16.cfg["groups"]["F"])


def existing(label, h):
    """A component of the flag's type already in the map at the flag (>= half of it inside the box grown 2 px,
    or under the flag point), nearest the point; None if there is none or the flag has no location. A flag from
    a symbol check (window / wall ink) names its parsed component itself (h["mask"])."""
    m = h.get("mask")
    if m is not None and h["type"] in MARKABLE + ["wall"] and (label[m] == I[h["type"]]).mean() >= 0.5:
        return m & (label == I[h["type"]])
    if h["type"] not in MARKABLE or h.get("point") is None:
        return None
    y0, x0, y1, x1 = h["box"]
    inbox = np.zeros(label.shape, bool)
    inbox[max(0, y0 - 2):y1 + 3, max(0, x0 - 2):x1 + 3] = True
    py, px = h["point"]
    found = [m for m in components(label == I[h["type"]]) if (m & inbox).sum() >= 0.5 * m.sum() or m[py, px]]
    return pick_one(found, h["point"])


def fill_hole(label, box, fc, generator, fast=None, typ=None, rules=(), point=None, gap=None):
    """point: (y, x) map px where the flag points; gap: bool map mask of a wall gap to close."""
    m = models()
    win, origin, hole = window_around(label, box, fc["window"])
    wpoint = None if point is None else (point[0] - origin[0], point[1] - origin[1])
    wgap = None if gap is None else window_around(gap.astype(np.uint8), box, fc["window"])[0] == 1
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
        sk = apply_prior(sketch[0].numpy().astype(np.uint8), hole > 0, typ, rules, wpoint, wgap)   # into the sketch
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


def pick_one(comps, point=None):
    """One component: the one nearest the flag point if there is one, else the largest (None if no component)."""
    if not comps:
        return None
    if point is None:
        return max(comps, key=lambda m: m.sum())
    py, px = point
    return min(comps, key=lambda m: (np.min((np.nonzero(m)[0] - py) ** 2 + (np.nonzero(m)[1] - px) ** 2), -m.sum()))


DOOR_WALL_PX = 2


def door_walls(changed, after, door, r=None):
    """The hybrid's new wall pixels within r px (default DOOR_WALL_PX) of a kept door: the wall it drew the door
    into (reverting them left the door floating beside the old wall line -> slivers, C7; user step 5, 2026-10-20)."""
    r = DOOR_WALL_PX if r is None else r
    near = cv2.dilate(door.astype(np.uint8), np.ones((2 * r + 1,) * 2, np.uint8)) > 0
    return changed & near & (after == I["wall"])


def interior_door(label, d):
    """Does the door join two interior spaces (room types on both faces)? Room holes never get a door on an
    exterior wall (user, 2026-10-20)."""
    pair = door_side_pair(label, d, R.CONST["T_wall"] + 2)
    return bool(pair) and all(c in C16.group("S") for c in pair)


def restrict(before, after, typ, point=None):
    """Type-restricted fill (user, chunk 8): inside a detected hole keep ONLY the detected type and
    revert every other changed pixel to the map before the fill.
      door / window / railing / a fixture  -> one new component (door / window: nearest the flag point,
                                              else the largest; a door clipped to one wall line at median
                                              length, plus the new wall pixels next to it)
      wall                                 -> new wall pixels
      room                                 -> new walls + at most one door (only between two interior
                                              spaces, never on an exterior wall) + ONE room type, the most
                                              common among the changed room pixels (user, 2026-10-20)
      manual                               -> everything the model drew
    A fixture is dropped if the room it lands in is not compatible with it (C15 table)."""
    if typ == "manual":
        return after
    changed = after != before
    if typ == "room":
        keep = changed & (after == I["wall"])
        doors = [d for d in components(changed & (after == I["door"])) if interior_door(after, d)]
        door = pick_one(doors, point)
        if door is not None:
            door = clip_door(door)
            keep |= door | door_walls(changed, after, door)
        rooms = changed & np.isin(after, C16.group("S"))
        if rooms.any():
            keep |= rooms & (after == int(np.bincount(after[rooms]).argmax()))
    elif typ == "wall":
        keep = changed & (after == I["wall"])
    else:
        one = pick_one(components(changed & (after == I[typ])), point if typ in ("door", "window") else None)
        keep = one if one is not None else np.zeros(after.shape, bool)
        if typ == "door" and keep.any():
            keep = clip_door(keep)
            keep = keep | door_walls(changed, after, keep)
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
            where = f"the {' / '.join(names)} wall" if names else "a wall"
            if "C8" in rules and rooms:
                lines.append(f"{rooms[0]} had no door -> door added on {where}")
            elif "R5" in rules and len(names) == 2:
                lines.append(f"open doorway between {names[0]} and {names[1]} had no door -> door added")
            else:
                lines.append(f"door added on {where}")
        for c in REAL:
            if c == "door":
                continue
            for m in components(changed & (after == I[c])):
                room = host_room(before, m) or "the plan"
                where = {"wall": f"next to {room}", "window": f"on the {room} wall"}.get(c, f"in {room}")
                lines.append(f"{c} added {where}")
        if h.get("gap") is not None and any(m.startswith("wall added") for m in lines):
            lines.insert(0, f"{h['gap_len']} px break in a wall line: no door fits -> wall missing, closed")
        if h["type"] in ("room", "region"):
            created = [C16.names[c] for c in np.unique(after[changed]) if c in C16.group("S")]
            if created:
                lines.append(f"missing room -> room drawn as {', '.join(created)}")
        if h.get("existing") is not None:
            lines = [f"{h['type']} symbol missing in the drawing -> added" if "WGAP" not in rules else
                     "window missing (a gap in the drawing) -> added"]
        if not lines:
            what = f"{h['type']} missing ({'+'.join(sorted(rules))})" if rules else "manual box"
            lines.append(f"{what} -> nothing added (the model kept the drawing as it was)")
        msgs += [m for m in lines if m not in msgs]
    return msgs


def overlay(image, before, after, scale, marked=None):
    """Additions (and `marked` parsed components: symbols missing in the drawing) in red on the original
    drawing; doors drawn as gap + leaf + quarter arc."""
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
    H, W = image.shape
    added = (after != before) if marked is None else ((after != before) | marked)
    added = added & ~np.isin(after, C16.group("S") + C16.group("OUT"))
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

def is_large(h, max_hole):
    """Region-missing flags (R8) and manual boxes larger than max_hole go to the large-hole fill."""
    y0, x0, y1, x1 = h["box"]
    return h["type"] == "region" or (h["type"] == "manual" and max(y1 - y0, x1 - x0) + 1 > max_hole)


def run_pipeline(image, boxes=None, seed=None, fast=False, large=None, t0_refine=None):
    """fast=True: the live-app settings (configs/stage2/pipeline.yaml `fast`). large: strategy for large holes
    ("big" / "coarse" / "tiled", src/stage2/complete/large_fill.py; default fill.large_strategy)."""
    c = cfg()
    large = large or c["fill"].get("large_strategy", "tiled")
    t0_refine = t0_refine or c["fill"].get("t0_refine", 150)
    times = {}
    t = time.time()
    parsed, scale, pinfo = parse(image, c["parse"])
    times["parse"] = time.time() - t
    t = time.time()
    if boxes is None:
        holes = [{"type": p.type, "rules": p.rules, "box": p.box, "score": p.score, "point": flag_point(p),
                  "mask": p.mask} for p in detect_holes(parsed, image)]
    else:
        holes = [{"type": "manual", "rules": [], "box": tuple(int(round(v * scale)) for v in b), "point": None}
                 for b in boxes]
    mh = c["fill"]["max_hole"]
    for h in holes:                                              # keep holes window-sized
        if is_large(h, mh):
            continue
        y0, x0, y1, x1 = h["box"]
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        h["box"] = (max(cy - mh // 2, y0), max(cx - mh // 2, x0), min(cy + mh // 2, y1), min(cx + mh // 2, x1))
        if h["point"] is not None:                               # the point stays inside the (cut) box
            y0, x0, y1, x1 = h["box"]
            h["point"] = (min(max(h["point"][0], y0), y1), min(max(h["point"][1], x0), x1))
    times["detect"] = time.time() - t
    t = time.time()
    g = torch.Generator().manual_seed(c["fill"]["seed"] if seed is None else seed)
    completed = parsed.copy()
    marked = np.zeros(parsed.shape, bool)
    for h in holes:
        ex = existing(completed, h)                              # first: a symbol already there is never a gap
        if ex is not None:                                       # already in the parse: report, draw, no fill
            h["existing"] = ex
            marked |= ex
            continue
        gap_rule(completed, h)
        if is_large(h, mh):                                      # region missing / big manual box
            y0, x0, y1, x1 = h["box"]
            hole = np.zeros(completed.shape, bool)
            hole[y0:y1 + 1, x0:x1 + 1] = True
            new, info = fill_large(completed, hole, large, c["fill"], c["fast"] if fast else None, g, t0_refine)
            kept = new
        else:
            new, info = fill_hole(completed, h["box"], c["fill"], g, c["fast"] if fast else None, h["type"],
                                  h["rules"], h["point"], h.get("gap"))
            kept = restrict(completed, new, h["type"], h["point"])
        y0, x0, y1, x1 = h["box"]                                # debug record: map before, raw hybrid output and
        sl = (slice(max(0, y0 - 16), y1 + 17), slice(max(0, x0 - 16), x1 + 17))   # what the restriction kept
        h["raw"] = {"origin": (sl[0].start, sl[1].start), "before": completed[sl].copy(), "hybrid": new[sl].copy(),
                    "kept": kept[sl].copy()}
        completed = kept
        h.update(info)
    times["fill"] = time.time() - t
    additions = {C16.names[k]: (completed != parsed) & (completed == k) for k in np.unique(completed[completed != parsed])}
    return {"parsed": parsed, "completed": completed, "holes": holes, "additions": additions, "marked": marked,
            "overlay": overlay(image, parsed, completed, scale, marked),
            "messages": describe_additions(parsed, completed, holes),
            "scale": scale, "parse_info": pinfo, "timings": {k: round(v, 1) for k, v in times.items()}}
