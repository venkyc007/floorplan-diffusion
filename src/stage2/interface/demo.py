"""Stage 2 tab of the app: run the pipeline on one drawing and turn the result into step-by-step pictures.

    from src.stage2.interface import demo
    grey = demo.to_grey(rgb_array)                    # any upload -> greyscale drawing (0 = ink, 255 = paper)
    steps = demo.run(grey)                            # automatic: PARSE -> CHECK -> FILL -> VERIFY
    steps = demo.run(grey, boxes=demo.mask_boxes(m))  # manual: the user marked the missing region -> PARSE + FILL

`run` returns a dict of display images (parsed map, what CHECK flagged, the filled plan, the additions in red on
the drawing) plus the pipeline's own messages, a per-hole table and timings. All model work is done by
src/stage2/pipeline/run.py run_pipeline (frozen settings, configs/stage2/pipeline.yaml); this module only
converts inputs and draws pictures. Gallery cases come from src/stage2/interface/make_gallery.py.
"""

import json

import cv2
import numpy as np
import yaml

from src.stage2.data.classes16 import C16
from src.stage2.pipeline.run import run_pipeline
from src.utils import PROJECT_ROOT

CFG = yaml.safe_load((PROJECT_ROOT / "configs" / "stage2" / "app.yaml").read_text(encoding="utf-8"))
GALLERY_DIR = PROJECT_ROOT / CFG["gallery_dir"]
AUTO_COLOUR = (220, 0, 0)        # CHECK flags (red)
USER_COLOUR = (0, 90, 220)       # boxes the user drew (blue)
ADDED_COLOUR = (255, 0, 200)     # outline of what the fill added (magenta)
MIN_MASK_PX = 30                 # brushed blobs smaller than this (image px) are ignored as stray clicks


class InputError(ValueError):
    """The uploaded image cannot be used (message shown to the user)."""


# ---------------------------------------------------------------- inputs

def to_grey(image):
    """Any Gradio image (H, W), (H, W, 3) or (H, W, 4), uint8 or float -> (H, W) uint8 drawing, ink dark on white.
    Transparent pixels count as paper; a drawing with a dark background (white lines) is inverted."""
    a = np.asarray(image)
    if a.dtype != np.uint8:
        a = np.clip(a * (255 if a.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 4:                       # composite on white paper
        alpha = a[..., 3:4].astype(np.float32) / 255
        a = (a[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    if a.ndim == 3:
        a = cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY)
    if np.median(a) < 128:                                    # mostly dark: white-on-black drawing
        a = 255 - a
    h, w = a.shape
    lim = CFG["upload"]
    if max(h, w) > lim["max_side_px"] or min(h, w) < lim["min_side_px"]:
        raise InputError(f"image is {w} x {h} px; use one between {lim['min_side_px']} and {lim['max_side_px']} px "
                         f"on each side")
    return a


def mask_boxes(mask, min_px=MIN_MASK_PX):
    """Brushed mask (H, W) bool in drawing px -> one box (y0, x0, y1, x1) per connected brushed region."""
    m = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8))   # strokes a few px apart = one region
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    boxes = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area >= min_px:
            boxes.append((int(y), int(x), int(y + h - 1), int(x + w - 1)))
    return boxes


# ---------------------------------------------------------------- gallery

def load_gallery():
    """Gallery index (list of case dicts with 'drawing' paths), or [] if it has not been built yet."""
    index = GALLERY_DIR / "index.json"
    if not index.exists():
        return []
    cases = json.loads(index.read_text(encoding="utf-8"))
    for c in cases:
        c["drawing"] = str(GALLERY_DIR / f"{c['id']:02d}_drawing.png")
    return cases


def case_caption(c):
    if c["kind"] == "intact":
        return "intact (nothing erased)"
    return f"{c['kind']} erased · {c['meta'].get('size_bin', '')}"


def load_case(c):
    """(erased drawing, original drawing, ground-truth labels, removed mask) of a gallery case."""
    read = lambda s, f=cv2.IMREAD_GRAYSCALE: cv2.imread(str(GALLERY_DIR / f"{c['id']:02d}_{s}.png"), f)
    return read("drawing"), read("original"), read("labels", cv2.IMREAD_UNCHANGED), read("removed") > 0


def box_on_drawing(grey, box, colour=(255, 140, 0), label=None):
    """The drawing as RGB with one box drawn (the gallery's 'where the part was erased', orange)."""
    rgb = cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)
    if box is not None:
        y0, x0, y1, x1 = box
        th = max(2, round(max(grey.shape) / 400))
        cv2.rectangle(rgb, (x0, y0), (x1, y1), colour, th)
        if label:
            cv2.putText(rgb, label, (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * th, colour, th)
    return fit(rgb)


# ---------------------------------------------------------------- pictures

def fit(img, side=None):
    """Resize for display: maps are enlarged with sharp pixels, big drawings shrunk smoothly."""
    side = side or CFG["display_max_side"]
    h, w = img.shape[:2]
    f = side / max(h, w)
    if f >= 2:
        k = int(f)
        return np.repeat(np.repeat(img, k, 0), k, 1)
    if f < 1:
        return cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
    return img


def label_image(label, side=None):
    """16-class label map -> RGB at display size (integer enlargement, so pixels stay sharp)."""
    rgb = C16.colourise(label)
    h, w = label.shape
    k = max(1, int((side or CFG["display_max_side"]) // max(h, w)))
    return np.repeat(np.repeat(rgb, k, 0), k, 1), k


def flags_image(parsed, holes):
    """Parsed map with every hole box: red = found by CHECK (type + which rule / detector), blue = user box."""
    rgb, k = label_image(parsed)
    for i, h in enumerate(holes, 1):
        y0, x0, y1, x1 = (v * k for v in h["box"])
        manual = h["type"] == "manual"
        colour = USER_COLOUR if manual else AUTO_COLOUR
        cv2.rectangle(rgb, (x0, y0), (x1 + k - 1, y1 + k - 1), colour, 2)
        text = f"{i}: your box" if manual else f"{i}: {h['type']} ({'+'.join(h['rules'])})"
        cv2.putText(rgb, text, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    return rgb


def completed_image(parsed, completed):
    """The completed map with a magenta outline around every changed component."""
    rgb, k = label_image(completed)
    changed = (completed != parsed).astype(np.uint8)
    if changed.any():
        big = np.repeat(np.repeat(changed, k, 0), k, 1)
        ring = cv2.dilate(big, np.ones((3, 3), np.uint8)) - big
        grown = cv2.dilate(ring, np.ones((2, 2), np.uint8)) > 0
        rgb[grown & (big == 0)] = ADDED_COLOUR
    return rgb


def hole_rows(holes):
    """One table row per hole: what was flagged, by what, and what the verifier counted."""
    rows = []
    for i, h in enumerate(holes, 1):
        if h.get("existing") is not None:
            what = "symbol already read by the parser: drawn in red, no fill"
        elif h["type"] == "manual":
            what = "filled (your box)"
        else:
            what = "filled"
        viol = h.get("violations")
        rows.append([i, h["type"], "+".join(h["rules"]) or "user", str(tuple(int(v) for v in h["box"])),
                     h.get("t0", ""), "" if viol is None else str(min(viol) if isinstance(viol, list) else viol), what])
    return rows


HOLE_HEADERS = ["#", "type", "found by", "box (map px)", "t0", "violations (best sample)", "action"]


def run(grey, boxes=None, seed=0, fast=None):
    """Run the pipeline (automatic if boxes is None, else manual on those drawing-px boxes) -> pictures + text."""
    fast = CFG["fast"] if fast is None else fast
    res = run_pipeline(grey, boxes=boxes, seed=seed, fast=fast)
    return {"res": res,
            "parsed": label_image(res["parsed"])[0],
            "flags": flags_image(res["parsed"], res["holes"]),
            "completed": completed_image(res["parsed"], res["completed"]),
            "overlay": fit(res["overlay"]),
            "messages": res["messages"],
            "rows": hole_rows(res["holes"]),
            "timings": res["timings"]}


def score(res, case):
    """For gallery cases (ground truth known): the validation's own scores (src/stage2/pipeline/e2e_val.score)."""
    from src.stage2.pipeline.e2e_val import score as e2e_score
    _, _, lbl, removed = load_case(case)
    if not removed.any():                                     # intact drawing: only false additions matter
        from src.stage2.pipeline.e2e_val import REAL
        from src.stage2.constraints.rules import components
        added = ((res["completed"] != res["parsed"]) | res["marked"]) & np.isin(res["completed"],
                                                                                 [C16.index[c] for c in REAL])
        return {"intact": True, "false_adds": len(components(added))}
    s = e2e_score(res, removed, lbl, case["expected_type"])
    return {"intact": False, **{k: s[k] for k in ("found", "right_type", "fixed", "fix_iou", "false_adds",
                                                  "tol_before", "tol_after", "valid_before", "valid_after")}}


def score_markdown(s, case):
    if s["intact"]:
        n = s["false_adds"]
        return (f"**Scored against the truth** (intact drawing): {n} false addition{'s' * (n != 1)} "
                f"({'none, as it should be' if n == 0 else 'the system changed a drawing that was complete'}).")
    yes = lambda b: "yes" if b else "no"
    return (f"**Scored against the truth** (the validation's definitions): erased {case['kind']} → "
            f"found **{yes(s['found'])}** (right type {yes(s['right_type'])}), fixed **{yes(s['fixed'])}** "
            f"(IoU {s['fix_iou']:.2f}; fixed = IoU ≥ 0.3), false additions elsewhere **{s['false_adds']}**, "
            f"tolerant mIoU in the region {s['tol_before']} → {s['tol_after']}, "
            f"valid {yes(s['valid_before'])} → {yes(s['valid_after'])}.")


def legend_html():
    chips = []
    for n, c in zip(C16.names, C16.palette):
        chips.append(f'<span style="display:inline-block;margin:0 10px 4px 0"><span style="display:inline-block;'
                     f'width:13px;height:13px;background:rgb({c[0]},{c[1]},{c[2]});border:1px solid #888;'
                     f'vertical-align:middle"></span> {n.replace("_", " ")}</span>')
    return "<div>" + "".join(chips) + "</div>"
