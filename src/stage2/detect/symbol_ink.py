"""Parse vs drawing double check for windows and walls (user, 2026-10-20): is the parsed component's symbol
actually drawn in the image? (Doors: src/stage2/detect/door_ink.py, unchanged.)

    feats = window_features(image, label)   # [(window mask, {"d_wall", "ink", "runs"})]
    runs  = wall_ink_runs(image, label)     # [(run mask, length px), ...], share of parsed wall px with no ink
    cands = window_ink_candidates(image, label, d_max, gap_ink)    # Candidate("window", ["WINK" | "WGAP"])
    cands = wall_ink_candidates(image, label, min_len)             # Candidate("wall", ["WALLINK"])

Completion-scale label (walls 3 px) mapped onto the drawing (any size); everything is measured in local crops.

WINDOW. Profiles ACROSS the wall line (one value per across offset, image px), over the window's parsed band
(+- 1 label px across, middle 70% along):
  grey profile  median grey of each across row
  d_wall        mean |grey profile of the window - grey profile of the parsed wall on the same line just past
                the window's ends| / 255. An erased window is the wall continued in the drawing's own style
                (erasure.paint_wall), so d_wall ~ 0; a drawn window (thin parallel lines on paper, or a light
                band) differs from the wall. NaN if no parsed wall continues the line (no reference).
  ink           share of stroke pixels (>= 20 grey levels darker than the local background) in the band
  runs          median number of dark runs (dark / light transitions / 2) across the band, the user's
                "thin parallel lines" signature; reported in the calibration, not used by the rule
  "window symbol missing (drawn as plain wall)": d_wall < d_max;  "window missing (gap)": ink < gap_ink.
WALL. A parsed wall pixel is "inked" if a stroke lies within 1 label px of it (stroke share per label pixel
from an area resize of the stroke mask, then a 3 x 3 max). Uninked wall pixels form runs (4-connected); a run of
>= min_len px (its longer side) = "wall missing" (the parser drew a wall over a gap).
Style guard (user): per drawing and type, if more than GUARD = 40% of its components (windows) / of its parsed
wall pixels (walls) lack their signature, the check is skipped there (a drawing style, not omissions).
"""

import cv2
import numpy as np

from src.stage2.constraints.rules import components
from src.stage2.data.classes16 import C16
from src.stage2.data.erasure import INK
from src.stage2.detect import rules as R

I = C16.index
GUARD = 0.4
MIN_WINDOW_PX = 6
DARK = 110           # absolute: thick solid walls (wider than the background kernel) are ink too


def _strokes(img):
    """Ink: >= INK grey levels darker than the local background (31 px closing), or darker than DARK."""
    bg = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    return ((bg.astype(np.int16) - img) > INK) | (img < DARK)


def window_features(image, label, min_px=MIN_WINDOW_PX):
    H, W = image.shape
    h, w = label.shape
    fy, fx = H / h, W / w
    wall = label == I["wall"]
    out = []
    for d in components(label == I["window"]):
        if d.sum() < min_px:
            continue
        ys, xs = np.nonzero(d)
        horiz = (xs.max() - xs.min()) >= (ys.max() - ys.min())
        # label-px extents: along [s0, s1], across [c0, c1] (+- 1 px)
        s0, s1 = (xs.min(), xs.max()) if horiz else (ys.min(), ys.max())
        c0, c1 = (ys.min() - 1, ys.max() + 1) if horiz else (xs.min() - 1, xs.max() + 1)
        L = s1 - s0 + 1
        f_al, f_ac = (fx, fy) if horiz else (fy, fx)
        # crop (image px) around the window and its wall neighbours on both ends
        a0, a1 = max(0, int((c0 - 3) * f_ac)), int((c1 + 4) * f_ac)
        b0, b1 = max(0, int((s0 - L - 2) * f_al)), int((s1 + L + 3) * f_al)
        img = image[a0:min(a1, H), b0:min(b1, W)] if horiz else image[b0:min(b1, H), a0:min(a1, W)].T
        if img.size == 0:
            continue
        dark = _strokes(img)
        ac = slice(max(0, int(c0 * f_ac) - a0), int((c1 + 1) * f_ac) - a0)        # across rows of the band

        def cols(p, q):                                                          # along columns (crop px)
            return max(0, int(p * f_al) - b0), max(0, int(q * f_al) - b0)
        m = max(1, int(0.15 * L))
        q0, q1 = cols(s0 + m, s1 + 1 - m)
        band, dband = img[ac, q0:q1], dark[ac, q0:q1]
        if band.size == 0 or band.shape[0] < 2:
            continue
        prof = np.median(band, axis=1)
        # parsed wall on the same line just past both ends (label px), mapped to crop columns
        mid = (c0 + c1) // 2
        ref = []
        for lo, hi in ((s0 - L, s0 - 1), (s1 + 1, s1 + L)):
            for s in range(max(0, lo), min(w if horiz else h, hi + 1)):
                if (wall[mid, s] if horiz else wall[s, mid]):
                    p, q = cols(s, s + 1)
                    ref.append(img[ac, p:max(q, p + 1)])
        ref = [r for r in ref if r.size and r.shape[0] == band.shape[0]]
        d_wall = float(np.mean(np.abs(prof - np.median(np.concatenate(ref, 1), axis=1))) / 255) if ref else np.nan
        runs = np.median([np.count_nonzero(np.diff(np.r_[0, col.astype(np.int8), 0]) == 1) for col in dband.T])
        out.append((d, {"d_wall": d_wall, "ink": float(dband.mean()), "runs": float(runs)}))
    return out


def window_verdict(f, d_max, gap_ink):
    """'wall' (drawn as plain wall), 'gap' (no ink) or None (window symbol drawn)."""
    if f["ink"] < gap_ink:
        return "gap"
    if np.isfinite(f["d_wall"]) and f["d_wall"] < d_max:
        return "wall"
    return None


def window_ink_candidates(image, label, d_max, gap_ink, guard=GUARD, feats=None):
    feats = window_features(image, label) if feats is None else feats
    v = [(d, window_verdict(f, d_max, gap_ink), f) for d, f in feats]
    if not v or np.mean([x is not None for _, x, _ in v]) > guard:
        return []
    out = []
    for d, x, f in v:
        if x is not None:
            ys, xs = np.nonzero(d)
            out.append(R.Candidate("window", R.box_of(d, label.shape), ["WINK" if x == "wall" else "WGAP"],
                                   1.0, (int(np.median(ys)), int(np.median(xs))), d))
    return out


def wall_ink_runs(image, label):
    """([(uninked run mask, length px)], share of parsed wall pixels with no ink nearby)."""
    h, w = label.shape
    s = cv2.resize(_strokes(image).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    inked = cv2.dilate(s, np.ones((3, 3), np.uint8)) > 0.02
    wall = label == I["wall"]
    bare = wall & ~inked
    runs = []
    for m in components(bare):
        ys, xs = np.nonzero(m)
        runs.append((m, int(max(ys.max() - ys.min(), xs.max() - xs.min()) + 1)))
    return runs, float(bare.sum() / max(wall.sum(), 1))


def wall_ink_candidates(image, label, min_len, guard=GUARD, runs=None):
    runs, share = wall_ink_runs(image, label) if runs is None else runs
    if share > guard:
        return []
    out = []
    for m, n in runs:
        if n >= min_len:
            ys, xs = np.nonzero(m)
            k = len(ys) // 2
            out.append(R.Candidate("wall", R.box_of(m, label.shape), ["WALLINK"], 1.0,
                                   (int(ys[k]), int(xs[k])), m))
    return out
