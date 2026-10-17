"""Draw a parsed CubiCasa plan as a 16-class label map, and make its greyscale image.

Label maps are always drawn straight from the SVG polygons at the target scale
(never by downsampling another label map: 3-px walls and doors would vanish).

Paint order (configs/stage2/classes16.yaml): background -> outdoor -> rooms ->
walls -> railings -> doors/windows -> fixtures. Fixtures only cover space /
outdoor / background pixels, never structure.

Coordinates: an SVG point (x, y) is a position in a continuous frame where
pixel i spans [i, i+1). After scaling by s, OpenCV's pixel-centre convention
needs a -0.5 shift; polygons are filled with 4 bits of sub-pixel precision.
"""

import cv2
import numpy as np

from src.stage2.data.classes16 import C16

SHIFT = 4  # sub-pixel bits for cv2.fillPoly


def frame_size(frame_hw, scale):
    h, w = frame_hw
    return max(1, int(round(h * scale))), max(1, int(round(w * scale)))


def _fill(label, pts, scale, value, allowed=None):
    p = np.round((pts * scale - 0.5) * (1 << SHIFT)).astype(np.int32)
    if allowed is None:
        cv2.fillPoly(label, [p], int(value), lineType=cv2.LINE_8, shift=SHIFT)
        return
    tmp = np.zeros(label.shape, np.uint8)
    cv2.fillPoly(tmp, [p], 1, lineType=cv2.LINE_8, shift=SHIFT)
    label[(tmp > 0) & allowed] = value


def draw_labels(plan, frame_hw, scale):
    """ParsedPlan -> (H, W) uint8 label map at `scale` x the F1_scaled frame.

    Also returns the map before fixtures were painted (the room under each
    fixture, used for the fixture-room co-occurrence statistics).
    """
    idx = C16.index
    label = np.full(frame_size(frame_hw, scale), idx["background"], np.uint8)
    spaces = plan.of_kind("space")
    for s in [s for s in spaces if s.cls == idx["outdoor"]] + [s for s in spaces if s.cls != idx["outdoor"]]:
        _fill(label, s.points, scale, s.cls)
    for kind in ("wall", "railing"):
        for s in plan.of_kind(kind):
            _fill(label, s.points, scale, s.cls)
    for s in plan.shapes:                       # doors and windows in document order
        if s.kind in ("door", "window"):
            _fill(label, s.points, scale, s.cls)
    before_fixtures = label.copy()
    structure = np.isin(label, C16.group("BOUNDARY"))
    overwrite = C16.cfg.get("fixtures_overwrite_structure", False)
    for s in plan.of_kind("fixture"):
        _fill(label, s.points, scale, s.cls, None if overwrite else ~structure)
    return label, before_fixtures


def fixture_footprint(plan, frame_hw, scale):
    """Boolean map of every fixture polygon (ignoring what lies underneath)."""
    mask = np.zeros(frame_size(frame_hw, scale), np.uint8)
    for s in plan.of_kind("fixture"):
        _fill(mask, s.points, scale, 1)
    return mask > 0


def load_grey(path):
    """Read a PNG as greyscale uint8; transparent pixels become white paper."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 2:
        return im
    if im.shape[2] == 4:
        alpha = im[:, :, 3:4].astype(np.float32) / 255
        im = (im[:, :, :3].astype(np.float32) * alpha + 255 * (1 - alpha)).astype(np.uint8)
    return cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2GRAY)


def make_image(grey, out_hw):
    """Resize a greyscale drawing to (H, W): area averaging when shrinking, cubic when enlarging."""
    h, w = out_hw
    shrink = h * w < grey.shape[0] * grey.shape[1]
    return cv2.resize(grey, (w, h), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC)
