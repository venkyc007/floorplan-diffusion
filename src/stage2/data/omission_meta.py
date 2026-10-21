"""Metadata of one image-level omission (user, 2026-10-20; deliverable 5), recorded with every generated case.

    meta = omission_meta(lbl, removed, "door (a)")
    -> {"type": "door (a)", "location": "exterior", "position": "edge", "size_bin": "small",
        "size_pct": 0.41, "center_dist": 0.73}

Fields (the Stage 1 rules, src/stage1/data/corruptions.py describe(), constants from configs/stage1/corruption.yaml):
  type       door (a) / door (b) / window / wall / fixture / room / railing / room (whole) / region
             (intact drawings: "intact")
  location   exterior if the omitted component touches outdoor / background (4-neighbours), else interior
  position   distance of the omission's centre from the building centre, divided by the building "radius"
             (largest centre-to-building-pixel distance): < 0.55 -> centre, else edge
  size_bin   omitted area as % of the building area: < 2.7 small, < 7.7 medium, else large
Building = every pixel that is not outdoor / background. `lbl` is the ground-truth label map at the same scale
as `removed` (the drawing's parser-scale labels).

breakdown(rows, field, cols) turns a list of per-case dicts into one markdown table per field, so every
evaluation table can be split by type, location, position and size bin.
"""

import cv2
import numpy as np
import yaml

from src.stage2.data.classes16 import C16
from src.utils import PROJECT_ROOT

_S1 = yaml.safe_load((PROJECT_ROOT / "configs" / "stage1" / "corruption.yaml").read_text(encoding="utf-8"))
EDGE_CUTOFF = _S1["position"]["edge_cutoff"]          # 0.55 (Stage 1)
SIZE_EDGES = tuple(_S1["size_bins"]["edges"])         # (2.7, 7.7) % of the building (Stage 1)
FIELDS = ("type", "location", "position", "size_bin")
ORDER = {"type": ["door (a)", "door (b)", "window", "wall", "railing", "fixture", "room", "room (whole)", "region",
                  "intact"],
         "location": ["interior", "exterior"], "position": ["centre", "edge"],
         "size_bin": ["small", "medium", "large"]}
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)


def omission_meta(lbl, removed, kind):
    """Metadata dict of one omission (see the module docstring). `removed`: bool mask, same shape as `lbl`."""
    if removed is None or not removed.any():
        return {"type": kind, "location": "", "position": "", "size_bin": "", "size_pct": "", "center_dist": ""}
    out = np.isin(lbl, C16.group("OUT"))
    building = ~out
    touches = (cv2.dilate(removed.astype(np.uint8), CROSS) > 0) & out & ~removed
    by, bx = np.nonzero(building)
    cy, cx = by.mean(), bx.mean()
    radius = max(float(np.sqrt((by - cy) ** 2 + (bx - cx) ** 2).max()), 1.0)
    my, mx = np.nonzero(removed)
    dist = float(np.hypot(my.mean() - cy, mx.mean() - cx) / radius)
    pct = 100.0 * removed.sum() / max(int(building.sum()), 1)
    small, large = SIZE_EDGES
    return {"type": kind,
            "location": "exterior" if touches.any() else "interior",
            "position": "edge" if dist >= EDGE_CUTOFF else "centre",
            "size_bin": "small" if pct < small else ("medium" if pct < large else "large"),
            "size_pct": round(float(pct), 3), "center_dist": round(dist, 3)}


def breakdown(rows, field, cols):
    """Markdown table: one line per value of `field` (fixed order), one column per entry of `cols`
    = {column name: function(list of rows) -> str}. Rows without that field are skipped."""
    vals = [v for v in ORDER.get(field, []) if any(r.get(field) == v for r in rows)]
    vals += sorted({r.get(field) for r in rows if r.get(field) not in vals and r.get(field) not in ("", None)})
    L = [f"| {field} | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for v in vals:
        sub = [r for r in rows if r.get(field) == v]
        L.append(f"| {v} | {len(sub)} | " + " | ".join(f(sub) for f in cols.values()) + " |")
    return L


def breakdowns(rows, cols, fields=FIELDS):
    """All four breakdown tables, each under a small heading."""
    L = []
    for f in fields:
        L += ["", f"By {f}:", ""] + breakdown(rows, f, cols)
    return L
