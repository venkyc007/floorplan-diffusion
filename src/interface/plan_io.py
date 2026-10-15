"""Turning user input (pictures, .npy files) into our 128x128 label maps.

The same steps as the data pipeline:
  1. colours -> classes: every pixel snaps to the nearest of our 6 class
     colours (src/data/classes.py); the grey "unknown" colour (how masked
     pixels are drawn) becomes UNKNOWN (it must then be repaired);
  2. framing on the building: the bounding box of walls + doors + windows is
     scaled so its longer side is 120 px (4 px margin), centred in 128x128,
     nearest-neighbour; anything that falls outside the frame is cropped,
     empty space is background.
Unusable input raises InputError with a message meant for the user.
"""

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.data.classes import BACKGROUND, COLORS, DOOR, NUM_CLASSES, WALL, WINDOW
from src.data.corruptions import UNKNOWN
from src.eval.figures import GREY

SIZE = 128
LONG_SIDE = 120                                   # the building's longer side after framing (4 px margin)
PALETTE = np.array([COLORS[c] for c in range(NUM_CLASSES)], dtype=float)   # (6, 3), class order


class InputError(ValueError):
    """Input that cannot be turned into a usable plan (message shown to the user)."""


# ------------------------------------------------------------- colours -> classes
def snap_colors(rgb, unknown_distance=12):
    """RGB (H, W, 3) -> (labels, distance to the chosen colour).
    Pixels within `unknown_distance` of the grey 'unknown' colour become UNKNOWN."""
    rgb = rgb[..., :3].astype(np.float32)
    labels = np.zeros(rgb.shape[:2], np.uint8)
    best = np.full(rgb.shape[:2], np.inf, np.float32)
    for c, color in enumerate(PALETTE):                  # one colour at a time: low memory for big images
        d = np.linalg.norm(rgb - color.astype(np.float32), axis=-1)
        closer = d < best
        labels[closer], best[closer] = c, d[closer]
    grey = np.linalg.norm(rgb - np.array(GREY, np.float32), axis=-1)
    unknown = grey <= unknown_distance
    labels[unknown] = UNKNOWN
    best[unknown] = grey[unknown]
    return labels, best


def block_majority(labels, scale):
    """Shrink a (H*scale, W*scale) label map to (H, W): the most common label of each block."""
    h, w = labels.shape[0] // scale, labels.shape[1] // scale
    blocks = labels[:h * scale, :w * scale].reshape(h, scale, w, scale).transpose(0, 2, 1, 3).reshape(h, w, -1)
    counts = np.stack([(blocks == v).sum(-1) for v in range(NUM_CLASSES)] + [(blocks == UNKNOWN).sum(-1)], -1)
    values = np.array(list(range(NUM_CLASSES)) + [UNKNOWN], np.uint8)
    return values[counts.argmax(-1)]


# ------------------------------------------------------------- framing
def structure_box(labels):
    """(r0, r1, c0, c1) of walls + doors + windows (inclusive), or None if there are none."""
    rows, cols = np.nonzero(np.isin(labels, [WALL, DOOR, WINDOW]))
    if len(rows) == 0:
        return None
    return rows.min(), rows.max(), cols.min(), cols.max()


def already_framed(labels, keep_long_side=(112, 124)):
    """True for a 128x128 map whose structure already spans about 120 px: it is left as is
    (re-sampling it would distort the 2-px walls)."""
    box = structure_box(labels)
    if labels.shape != (SIZE, SIZE) or box is None:
        return False
    long_side = max(box[1] - box[0], box[3] - box[2]) + 1
    return keep_long_side[0] <= long_side <= keep_long_side[1]


def frame_on_building(labels):
    """Scale + centre so the structure's bounding box has a longer side of 120 px (nearest neighbour)."""
    box = structure_box(labels)
    if box is None:
        raise InputError("No walls, doors or windows found, so the building cannot be framed. "
                         "Draw or upload a plan with walls in our wall colour (near-black).")
    r0, r1, c0, c1 = box
    h, w = r1 - r0 + 1, c1 - c0 + 1
    s = LONG_SIDE / max(h, w)                                     # output px per input px
    off_r, off_c = (SIZE - h * s) / 2, (SIZE - w * s) / 2         # where the box starts in the output
    out_r, out_c = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing="ij")
    src_r = np.floor(r0 + (out_r + 0.5 - off_r) / s).astype(int)  # input pixel under each output pixel centre
    src_c = np.floor(c0 + (out_c + 0.5 - off_c) / s).astype(int)
    inside = (src_r >= 0) & (src_r < labels.shape[0]) & (src_c >= 0) & (src_c < labels.shape[1])
    out = np.full((SIZE, SIZE), BACKGROUND, np.uint8)
    out[inside] = labels[src_r[inside], src_c[inside]]
    return out


def to_plan(labels, cfg):
    """Frame (unless already in our frame) and check that the result is a usable plan."""
    plan = labels if already_framed(labels, cfg["keep_frame_long_side"]) else frame_on_building(labels)
    walls = int((plan == WALL).sum())
    if walls < cfg["min_wall_px"]:
        raise InputError(f"Only {walls} wall pixels after framing (at least {cfg['min_wall_px']} needed). "
                         "This does not look like a floor plan in our colours.")
    return plan


# ------------------------------------------------------------- files
def load_png(data, cfg):
    """PNG/JPEG bytes in our class colours -> 128x128 label map."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise InputError(f"Not a readable image ({type(e).__name__}). Upload a PNG in our 6 class colours "
                         "or a .npy label map.") from None
    if max(img.size) > cfg["max_side_px"] or min(img.size) < 16:
        raise InputError(f"Image is {img.size[0]}x{img.size[1]} px; it must be 16-{cfg['max_side_px']} px per side.")
    rgba = np.asarray(img.convert("RGBA"))
    rgb = rgba[..., :3].copy()
    rgb[rgba[..., 3] < 128] = COLORS[BACKGROUND]                 # transparent = background
    labels, dist = snap_colors(rgb, cfg["unknown_distance"])
    off = 100 * (dist > cfg["off_palette_distance"]).mean()
    if off > cfg["max_off_palette_pct"]:
        raise InputError(f"{off:.0f}% of the pixels are far from every class colour (at most "
                         f"{cfg['max_off_palette_pct']}% allowed). Draw the plan in our 6 colours: wall "
                         "near-black, door red, window blue, room sand, outdoor green, background white.")
    return to_plan(labels, cfg)


def load_npy(data, cfg):
    """.npy bytes: a 2-D label map (values 0-5, 255 = unknown) or a (6, H, W) one-hot/score array."""
    try:
        arr = np.load(io.BytesIO(data), allow_pickle=False)       # never unpickle user files
    except Exception as e:                                         # noqa: BLE001 - any parse failure
        raise InputError(f"Not a readable .npy file ({type(e).__name__}).") from None
    if arr.ndim == 3 and arr.shape[0] == NUM_CLASSES:
        arr = arr.argmax(0)
    if arr.ndim != 2:
        raise InputError(f"Expected a 2-D label map (H, W) or a ({NUM_CLASSES}, H, W) array; got shape {arr.shape}.")
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.mod(arr, 1) == 0):
            raise InputError(f"Label maps must hold whole numbers (class ids); got {arr.dtype} values.")
    if max(arr.shape) > cfg["max_side_px"] or min(arr.shape) < 16:
        raise InputError(f"Label map is {arr.shape[0]}x{arr.shape[1]}; it must be 16-{cfg['max_side_px']} per side.")
    bad = np.setdiff1d(np.unique(arr), list(range(NUM_CLASSES)) + [UNKNOWN])
    if len(bad):
        raise InputError(f"Unknown class ids {bad[:5].tolist()}: allowed are 0-5 "
                         f"(wall, door, window, room, outdoor, background) and {UNKNOWN} (unknown).")
    return to_plan(arr.astype(np.uint8), cfg)


def load_file(path, cfg):
    """Dispatch on the file name; returns a 128x128 label map or raises InputError."""
    name = str(path).lower()
    with open(path, "rb") as f:
        data = f.read()
    if name.endswith(".npy"):
        return load_npy(data, cfg)
    if name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        return load_png(data, cfg)
    raise InputError("Unsupported file type: upload a .png image in our class colours or a .npy label map.")
