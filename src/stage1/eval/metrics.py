"""Evaluation metrics for floor plan inpainting.

Every metric compares a predicted label map with the ground truth, focused
on the masked region (the part the model had to fill in).

Before scoring, the prediction OUTSIDE the mask is replaced by the known
pixels (`composite`): the model was given them, so it is neither rewarded
nor punished there.

Metrics (definitions + reasons in docs/decisions.md):
  1. Per-class IoU inside the mask (wall, door, window, room, outdoor),
     mIoU (per sample, classes absent from both are skipped), pixel accuracy.
     Also a TOLERANT IoU for the thin classes (wall, door, window): a pixel
     counts as matched if the other map has the same class within 1 px
     (as in dedup), and a tolerant mIoU (tolerant for wall/door/window,
     strict for room/outdoor).
  2. Component detection precision / recall / F1 for door, window, room:
     4-connected components overlapping the mask, matched one-to-one
     (Hungarian) when their IoU >= 0.5.
  3. Structural metrics (boundary = wall + door + window):
       conn_err        |#boundary components pred - gt| on the whole plan (8-connectivity)
       open_on_wall    share of predicted doors/windows in the mask that touch a wall
       room_leak       share of predicted rooms in/next to the mask that touch outdoor/background
       room_count_err  |#enclosed regions pred - gt| on the whole plan: regions =
                       4-connected components of non-boundary pixels that do not
                       touch the image edge (the edge-touching ones are outside)
       alignment       share of predicted boundary pixels in the mask that lie on a
                       straight horizontal/vertical run of >= ALIGN_RUN pixels

Main entry points:
  sample_stats(pred, gt, mask)        raw counts for one sample
  summarize(list_of_stats)            turn counts into the metric values
  evaluate_benchmark(pred_fn, ...)    run a model over a benchmark, broken down
                                      by type / location / position / size bin
"""

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.stage1.data.classes import (BACKGROUND, CLASSES, DOOR, OUTDOOR, ROOM, WALL,
                              WINDOW)
from src.stage1.data.corruptions import to_model_input

IOU_CLASSES = [WALL, DOOR, WINDOW, ROOM, OUTDOOR]
DETECT_CLASSES = [DOOR, WINDOW, ROOM]
MATCH_IOU = 0.5
ALIGN_RUN = 5
TOLERANT_CLASSES = [WALL, DOOR, WINDOW]
TOLERANCE = 1   # px, for the tolerant IoU
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)


# ------------------------------------------------------------- helpers

def composite(pred, gt, mask):
    """Prediction inside the mask, known ground truth outside it."""
    return np.where(mask, pred, gt)


def label_components(binary, connectivity=4):
    """(number of components, component image) of a True/False mask."""
    n, lab = cv2.connectedComponents(binary.astype(np.uint8), connectivity=connectivity)
    return n - 1, lab


def components_in_mask(binary, mask):
    """List of component masks (4-connectivity) that overlap the mask."""
    n, lab = label_components(binary)
    ids = np.unique(lab[mask & binary])
    return [lab == i for i in ids if i > 0]


def touches(part, label, classes):
    """True if a pixel next to `part` (4-adjacency) has one of `classes`."""
    grown = cv2.dilate(part.astype(np.uint8), CROSS).astype(bool)
    return bool(np.isin(label[grown & ~part], classes).any())


def match_components(pred_parts, gt_parts):
    """One-to-one matching with IoU >= MATCH_IOU. Returns (tp, fp, fn)."""
    if not pred_parts or not gt_parts:
        return 0, len(pred_parts), len(gt_parts)
    iou = np.zeros((len(pred_parts), len(gt_parts)))
    for i, p in enumerate(pred_parts):
        for j, g in enumerate(gt_parts):
            inter = np.count_nonzero(p & g)
            if inter:
                iou[i, j] = inter / np.count_nonzero(p | g)
    rows, cols = linear_sum_assignment(-iou)  # Hungarian: maximise total IoU
    tp = int((iou[rows, cols] >= MATCH_IOU).sum())
    return tp, len(pred_parts) - tp, len(gt_parts) - tp


def enclosed_regions(label):
    """Number of regions enclosed by the boundary (wall + door + window).

    Non-boundary pixels are split into 4-connected regions; regions touching
    the image edge are the outside world (yards, open balconies, background)
    and are not counted. A gap in a wall (e.g. a missing door left empty)
    merges the two rooms on either side into one region.
    """
    n, lab = label_components(~np.isin(label, [WALL, DOOR, WINDOW]))
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    return n - int((edge > 0).sum())


def tolerant_counts(pred, gt, mask, cls):
    """(matches, total) for the tolerant IoU of one class inside the mask.

    A predicted pixel matches if the ground truth has the class within
    TOLERANCE px, and vice versa; matches = average of the two directions.
    Tolerant IoU = matches / (total - matches); equals strict IoU at tolerance 0.
    Only pixels INSIDE the mask are compared on both sides; otherwise known
    walls just outside the mask would "match" true walls at the mask edge
    and give credit for pixels the model never drew.
    """
    kernel = np.ones((2 * TOLERANCE + 1,) * 2, np.uint8)
    p_in = ((pred == cls) & mask).astype(np.uint8)
    g_in = ((gt == cls) & mask).astype(np.uint8)
    matches = (np.count_nonzero(p_in & cv2.dilate(g_in, kernel)) +
               np.count_nonzero(g_in & cv2.dilate(p_in, kernel))) / 2
    return matches, int(p_in.sum()) + int(g_in.sum())


def aligned_pixels(boundary):
    """Boundary pixels lying on a horizontal or vertical run of >= ALIGN_RUN px."""
    b = boundary.astype(np.uint8)
    horiz = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((1, ALIGN_RUN), np.uint8))
    vert = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((ALIGN_RUN, 1), np.uint8))
    return (horiz | vert).astype(bool)


# ------------------------------------------------------------ per sample

def sample_stats(pred, gt, mask):
    """Raw counts for one sample (pred, gt: label maps; mask: bool)."""
    pred = composite(pred, gt, mask)
    s = {"mask_px": int(mask.sum()),
         "correct_px": int((pred[mask] == gt[mask]).sum())}

    # 1. IoU inside the mask
    ious, tolerant_ious = [], []
    for c in IOU_CLASSES:
        p, g = (pred == c) & mask, (gt == c) & mask
        inter, union = int((p & g).sum()), int((p | g).sum())
        s[f"inter_{CLASSES[c]}"], s[f"union_{CLASSES[c]}"] = inter, union
        if not union:
            continue
        ious.append(inter / union)
        if c in TOLERANT_CLASSES:
            matches, total = tolerant_counts(pred, gt, mask, c)
            s[f"tmatch_{CLASSES[c]}"], s[f"ttotal_{CLASSES[c]}"] = matches, total
            tolerant_ious.append(matches / (total - matches))
        else:
            tolerant_ious.append(inter / union)
    s["miou"] = float(np.mean(ious)) if ious else np.nan
    s["tmiou"] = float(np.mean(tolerant_ious)) if tolerant_ious else np.nan

    # 2. Component detection
    for c in DETECT_CLASSES:
        tp, fp, fn = match_components(components_in_mask(pred == c, mask),
                                      components_in_mask(gt == c, mask))
        s[f"tp_{CLASSES[c]}"], s[f"fp_{CLASSES[c]}"], s[f"fn_{CLASSES[c]}"] = tp, fp, fn

    # 3. Structure
    boundary_p = np.isin(pred, [WALL, DOOR, WINDOW])
    boundary_g = np.isin(gt, [WALL, DOOR, WINDOW])
    s["conn_err"] = abs(label_components(boundary_p, 8)[0] - label_components(boundary_g, 8)[0])

    openings = components_in_mask(pred == DOOR, mask) + components_in_mask(pred == WINDOW, mask)
    s["n_openings"] = len(openings)
    s["n_openings_on_wall"] = sum(touches(o, pred, [WALL]) for o in openings)

    # Rooms overlapping OR bordering the mask: a missing wall in the mask
    # opens the neighbouring rooms even if no room pixel is drawn inside it.
    near_mask = cv2.dilate(mask.astype(np.uint8), CROSS).astype(bool)
    rooms = components_in_mask(pred == ROOM, near_mask)
    s["n_rooms_in_mask"] = len(rooms)
    s["n_rooms_leaking"] = sum(touches(r, pred, [OUTDOOR, BACKGROUND]) for r in rooms)
    s["room_count_err"] = abs(enclosed_regions(pred) - enclosed_regions(gt))

    s["boundary_px"] = int((boundary_p & mask).sum())
    s["aligned_px"] = int((aligned_pixels(boundary_p) & mask).sum())
    return s


# ------------------------------------------------------------ aggregation

def _ratio(a, b):
    return a / b if b else np.nan


def summarize(stats):
    """Metric values for a group of samples (a DataFrame or list of sample_stats).

    Pixel counts and TP/FP/FN are SUMMED over the group before dividing
    (so big masks / many components weigh more); conn_err, room_count_err
    and mIoU are averaged per sample.
    """
    d = pd.DataFrame(stats)
    t = d.sum(numeric_only=True)
    out = {"n": len(d), "mIoU": d["miou"].mean(), "tmIoU": d["tmiou"].mean()}
    for c in IOU_CLASSES:
        name = CLASSES[c]
        out[f"IoU_{name}"] = _ratio(t[f"inter_{name}"], t[f"union_{name}"])
    for c in TOLERANT_CLASSES:
        name = CLASSES[c]
        matches = t.get(f"tmatch_{name}", 0)
        out[f"tIoU_{name}"] = _ratio(matches, t.get(f"ttotal_{name}", 0) - matches)
    out["pixel_acc"] = _ratio(t["correct_px"], t["mask_px"])
    for c in DETECT_CLASSES:
        name = CLASSES[c]
        tp, fp, fn = t[f"tp_{name}"], t[f"fp_{name}"], t[f"fn_{name}"]
        p, r = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
        out[f"P_{name}"], out[f"R_{name}"] = p, r
        out[f"F1_{name}"] = _ratio(2 * tp, 2 * tp + fp + fn)
    out["conn_err"] = d["conn_err"].mean()
    out["open_on_wall"] = _ratio(t["n_openings_on_wall"], t["n_openings"])
    out["room_leak"] = _ratio(t["n_rooms_leaking"], t["n_rooms_in_mask"])
    out["room_count_err"] = d["room_count_err"].mean()
    out["alignment"] = _ratio(t["aligned_px"], t["boundary_px"])
    return out


# ------------------------------------------------------------- benchmarks

BREAKDOWNS = ["type", "location", "position", "size_bin"]


def load_benchmark(path):
    """Load <split>_benchmark.npz + its .csv metadata."""
    z = np.load(path)
    meta = pd.read_csv(str(path).replace(".npz", ".csv"))
    return {k: z[k] for k in z.files}, meta


def evaluate_benchmark(pred_fn, bench, meta, batch_size=64):
    """Run pred_fn over a whole benchmark and score it.

    pred_fn(batch) gets a dict with
        "inputs"    (B, 7, H, W) float32  model input (6 one-hot ch + mask)
        "corrupted" (B, H, W) uint8       label map, masked pixels = 255
        "masks"     (B, H, W) bool
    and must return predicted label maps (B, H, W).

    Returns (table, per_sample): table has one row "overall" plus one row per
    value of type / location / position / size_bin.
    """
    per_sample = []
    n = len(bench["masks"])
    for s in range(0, n, batch_size):
        corrupted, masks = bench["inputs"][s:s + batch_size], bench["masks"][s:s + batch_size]
        batch = {"corrupted": corrupted, "masks": masks,
                 "inputs": np.stack([to_model_input(c, m) for c, m in zip(corrupted, masks)])}
        preds = pred_fn(batch)
        for k, pred in enumerate(preds):
            per_sample.append(sample_stats(pred, bench["targets"][s + k], masks[k]))
    per_sample = pd.concat([meta.reset_index(drop=True), pd.DataFrame(per_sample)], axis=1)

    rows = [{"group": "overall", **summarize(per_sample)}]
    for col in BREAKDOWNS:
        for value, part in per_sample.groupby(col, sort=True):
            rows.append({"group": f"{col}={value}", **summarize(part)})
    return pd.DataFrame(rows).set_index("group"), per_sample


def to_markdown(table, digits=3):
    """A DataFrame as a GitHub markdown table (no extra package needed)."""
    cols = list(table.columns)
    lines = ["| " + " | ".join([table.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in table.iterrows():
        cells = [str(int(v)) if col == "n" else f"{v:.{digits}f}" if isinstance(v, float) else str(v)
                 for col, v in row.items()]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)
