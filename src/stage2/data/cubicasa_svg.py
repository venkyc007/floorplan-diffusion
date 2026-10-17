"""Read one CubiCasa5K `model.svg` into a list of labelled polygons.

Follows the conventions of CubiCasa's own loader (floortrans/loaders/house.py):
  * SVG coordinates are pixels of the plan's `F1_scaled.png` (the SVG's own
    width/height attributes are NOT the image frame);
  * walls / railings / windows / doors are <g id="Wall|Railing|Window|Door">
    whose first <polygon> child is the shape (no transform applied);
  * rooms are <g class="Space <Type> ...">, same polygon rule;
  * fixtures are <g class="FixedFurniture <Name> ...">: the polygon of their
    "BoundaryPolygon" child (or the bounding box of their drawing), mapped
    through the element's transform and, inside a "FixedFurnitureSet", the
    parent's transform too.

The output is plain data (numpy point arrays), so drawing it at any scale is a
separate step (rasterise.py).
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from src.stage2.data.classes16 import C16

SVG_NS = "{http://www.w3.org/2000/svg}"
NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass
class Shape:
    kind: str            # wall | railing | window | door | space | fixture
    cls: int             # Stage 2 class index (C16)
    name: str            # raw CubiCasa name (e.g. "Bedroom", "Toilet", "Wall External")
    points: np.ndarray   # (N, 2) float x, y in F1_scaled pixel coordinates


@dataclass
class ParsedPlan:
    shapes: list = field(default_factory=list)
    default_spaces: dict = field(default_factory=dict)     # Space types sent to space_default
    ignored_fixtures: dict = field(default_factory=dict)   # fixture names not painted
    n_bad: int = 0                                          # elements without usable geometry

    def of_kind(self, kind):
        return [s for s in self.shapes if s.kind == kind]


def _tag(el):
    return el.tag.replace(SVG_NS, "")


def parse_points(text):
    """'x1,y1 x2,y2 ...' -> (N, 2) float array (tolerates any separators)."""
    vals = [float(v) for v in NUM.findall(text or "")]
    return np.array(vals[: len(vals) // 2 * 2], dtype=float).reshape(-1, 2)


def parse_transform(text):
    """SVG transform attribute -> 3x3 matrix (matrix/translate/scale/rotate, composed left to right)."""
    m = np.eye(3)
    for op, args in re.findall(r"(\w+)\s*\(([^)]*)\)", text or ""):
        a = [float(v) for v in NUM.findall(args)]
        if op == "matrix" and len(a) == 6:
            t = np.array([[a[0], a[2], a[4]], [a[1], a[3], a[5]], [0, 0, 1]])
        elif op == "translate":
            t = np.array([[1, 0, a[0]], [0, 1, a[1] if len(a) > 1 else 0], [0, 0, 1]])
        elif op == "scale":
            sy = a[1] if len(a) > 1 else a[0]
            t = np.diag([a[0], sy, 1.0])
        elif op == "rotate":
            r = np.deg2rad(a[0])
            t = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
            if len(a) == 3:
                c = np.array([[1, 0, a[1]], [0, 1, a[2]], [0, 0, 1]])
                t = c @ t @ np.linalg.inv(c)
        else:
            continue
        m = m @ t
    return m


def apply(m, pts):
    h = np.c_[pts, np.ones(len(pts))] @ m.T
    return h[:, :2]


def first_polygon(el):
    for ch in el:
        if _tag(ch) == "polygon":
            return parse_points(ch.get("points"))
    return None


def all_points(el):
    """Every coordinate drawn inside an element (polygons, rects, lines, paths), local frame."""
    pts = []
    for d in el.iter():
        t = _tag(d)
        if t in ("polygon", "polyline"):
            pts.append(parse_points(d.get("points")))
        elif t == "rect":
            x, y = float(d.get("x", 0)), float(d.get("y", 0))
            w, h = float(d.get("width", 0)), float(d.get("height", 0))
            pts.append(np.array([[x, y], [x + w, y + h]]))
        elif t == "line":
            pts.append(np.array([[float(d.get("x1", 0)), float(d.get("y1", 0))],
                                 [float(d.get("x2", 0)), float(d.get("y2", 0))]]))
        elif t == "circle":
            cx, cy, r = (float(d.get(k, 0)) for k in ("cx", "cy", "r"))
            pts.append(np.array([[cx - r, cy - r], [cx + r, cy + r]]))
    pts = [p for p in pts if len(p)]
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def bbox_polygon(pts):
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def fixture_polygon(el, parent):
    """Fixture outline in image coordinates (CubiCasa's get_icon logic, simplified)."""
    boundary = next((ch for ch in el if _tag(ch) == "g" and ch.get("class") == "BoundaryPolygon"), None)
    local = first_polygon(boundary) if boundary is not None else None
    if local is None or len(local) < 3:
        pts = all_points(boundary if boundary is not None else el)
        if len(pts) < 2:
            return None
        local = bbox_polygon(pts)
    m = parse_transform(el.get("transform"))
    if parent is not None and parent.get("class") == "FixedFurnitureSet":
        m = parse_transform(parent.get("transform")) @ m
    return apply(m, local)


def parse_svg(path_or_text):
    """Parse a CubiCasa model.svg (path or SVG text) into a ParsedPlan."""
    text = path_or_text
    if not str(path_or_text).lstrip().startswith("<"):
        with open(path_or_text, encoding="utf-8") as f:
            text = f.read()
    root = ET.fromstring(text)
    parent_of = {ch: p for p in root.iter() for ch in p}
    plan = ParsedPlan()
    idx = C16.index
    for el in root.iter():
        if _tag(el) != "g":
            continue
        gid, cls_attr = el.get("id", ""), el.get("class", "") or ""
        words = cls_attr.split()
        kind = None
        if gid in ("Wall", "Railing", "Window", "Door"):
            kind = gid.lower()
            cls = {"wall": idx["wall"], "railing": idx["railing"],
                   "window": idx["window"], "door": idx["door"]}[kind]
            name = cls_attr.strip() or gid
            pts = first_polygon(el)
        elif words[:1] == ["Space"]:
            kind = "space"
            name = words[1] if len(words) > 1 else "Undefined"
            if name not in C16.cfg["spaces"]:
                plan.default_spaces[name] = plan.default_spaces.get(name, 0) + 1
            cls = C16.space_class(name)
            pts = first_polygon(el)
        elif words[:1] == ["FixedFurniture"]:
            name = " ".join(words[1:]) or "Misc"
            cls = C16.fixture_class(words[1:])
            if cls is None:
                plan.ignored_fixtures[name] = plan.ignored_fixtures.get(name, 0) + 1
                continue
            kind = "fixture"
            pts = fixture_polygon(el, parent_of.get(el))
        if kind is None:
            continue
        if pts is None or len(pts) < 3 or polygon_area(pts) <= 0:
            plan.n_bad += 1
            continue
        plan.shapes.append(Shape(kind, cls, name, pts))
    return plan


# ---------------------------------------------------------------- geometry helpers

def polygon_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_perimeter(pts):
    return float(np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1).sum())


def thickness_length(pts):
    """(thickness, length) of a wall-like polygon, exact for rectangles.

    A rectangle with sides L >= t has area A = L t and half-perimeter p = L + t,
    so t and L are the roots of x^2 - p x + A = 0. For other shapes (mitred
    ends, L-shapes) this is a close approximation. Returns (nan, nan) if degenerate.
    """
    a, p = polygon_area(pts), polygon_perimeter(pts) / 2
    disc = p * p - 4 * a
    if a <= 0 or disc < 0:
        return float("nan"), float("nan")
    t = (p - np.sqrt(disc)) / 2
    return float(t), float(a / t) if t > 0 else float("nan")


def wall_thickness(plan, min_aspect=2.0):
    """Length-weighted median thickness of the plan's wall polygons (F1_scaled px).

    Only elongated pieces (length >= min_aspect x thickness) count; short
    square-ish corner pieces have no meaningful thickness. Returns (median, n_walls_used).
    """
    tl = np.array([thickness_length(s.points) for s in plan.of_kind("wall")]).reshape(-1, 2)
    ok = np.isfinite(tl).all(1) & (tl[:, 0] > 0) & (tl[:, 1] >= min_aspect * tl[:, 0])
    if not ok.any():
        return float("nan"), 0
    t, w = tl[ok, 0], tl[ok, 1]
    order = np.argsort(t)
    cum = np.cumsum(w[order])
    return float(t[order][np.searchsorted(cum, cum[-1] / 2)]), int(ok.sum())


def opening_length(pts):
    """Length (longer side) of a door/window polygon, via its thickness/length."""
    t, length = thickness_length(pts)
    return length
