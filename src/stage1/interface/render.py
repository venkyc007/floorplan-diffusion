"""Pictures for the interface: label maps <-> screen images (128 px shown at 4x)."""

import numpy as np

from src.stage1.data.classes import DOOR, WALL, WINDOW
from src.stage1.data.corruptions import UNKNOWN
from src.stage1.eval.figures import colorize
from src.stage1.interface.plan_io import block_majority, snap_colors

ADDED = (255, 0, 255)       # magenta: wall/door/window pixels the repair drew
REMOVED = (0, 200, 255)     # cyan: wall/door/window pixels that were there before and are gone
CHANGED = (255, 200, 0)     # amber: other pixels whose class changed
MASK_RGBA = (150, 150, 150, 200)   # how brushed mask pixels look (our grey)


def upscale(img, scale):
    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)


def plan_image(label, scale, mask=None):
    """Colour picture of a label map; masked pixels grey."""
    shown = label.copy()
    if mask is not None:
        shown[mask] = UNKNOWN
    return upscale(colorize(shown.astype(np.uint8)), scale)


def editor_to_labels(rgb, scale, unknown_distance=12):
    """Screen picture from the plan editor -> 128x128 label map (colour snap + block majority)."""
    labels, _ = snap_colors(rgb, unknown_distance)
    return block_majority(labels, scale)


def mask_layer(mask, scale):
    """Transparent RGBA layer with the mask painted grey (for the mask editor)."""
    layer = np.zeros(mask.shape + (4,), np.uint8)
    layer[mask] = MASK_RGBA
    return upscale(layer, scale)


def layers_to_mask(layers, scale, coverage=0.25):
    """Brush layers (RGBA, screen size) -> 128x128 bool mask: a pixel is masked if at least
    `coverage` of its screen block is painted."""
    painted = None
    for layer in layers or []:
        if layer is None:
            continue
        layer = np.asarray(layer)
        alpha = layer[..., 3] > 0 if layer.ndim == 3 and layer.shape[-1] == 4 else layer.any(-1)
        painted = alpha if painted is None else painted | alpha
    if painted is None:
        return None
    h, w = painted.shape[0] // scale, painted.shape[1] // scale
    share = painted[:h * scale, :w * scale].reshape(h, scale, w, scale).mean((1, 3))
    return share >= coverage


def difference_image(before, after, mask, scale):
    """Repaired plan faded, with what the repair changed inside the mask highlighted:
    magenta = wall/door/window drawn by the repair, cyan = wall/door/window removed (only where the
    input was known), amber = other class changes where the input was known."""
    img = colorize(after.astype(np.uint8)).astype(float)
    img = (0.35 * img + 0.65 * 255).astype(np.uint8)              # fade everything
    img[~mask] = (0.6 * img[~mask] + 0.4 * 200).astype(np.uint8)  # outside the hole even fainter
    structure = [WALL, DOOR, WINDOW]
    known = before != UNKNOWN
    was = np.isin(before, structure) & known
    now = np.isin(after, structure)
    img[mask & known & (before != after) & ~was & ~now] = CHANGED
    img[mask & was & ~now] = REMOVED
    img[mask & now & ~was] = ADDED
    return upscale(img, scale)
