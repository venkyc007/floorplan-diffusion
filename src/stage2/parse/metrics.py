"""Parser validation metrics on whole drawings.

  per-class IoU   pooled over all val drawings (sum of intersections / sum of unions);
                  pixels labelled 255 (ignore) are skipped; mIoU = mean over classes
                  that occur in the val set.
  component P/R/F1 for door, window, toilet, sink, bath_shower, appliance: 4-connected
                  components of prediction and truth, matched one-to-one by the Hungarian
                  method on IoU; a match counts if IoU >= 0.5; TP/FP/FN summed over drawings.
  fallback        the user's trigger for CubiCasa's pretrained model: val wall IoU < 0.5
                  or door / window component F1 < 0.3.
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.stage2.data.classes16 import C16


class ParserMetrics:
    def __init__(self, component_classes, iou_thr=0.5, ignore_index=255):
        self.n = C16.n
        self.conf = np.zeros((self.n, self.n), np.int64)
        self.comp_classes = [C16.index[c] for c in component_classes]
        self.counts = {c: [0, 0, 0] for c in self.comp_classes}      # TP, FP, FN
        self.thr, self.ignore = iou_thr, ignore_index

    def update(self, pred, gt):
        """pred, gt: (H, W) uint8/int class maps of one drawing."""
        valid = gt != self.ignore
        idx = gt[valid].astype(np.int64) * self.n + pred[valid].astype(np.int64)
        self.conf += np.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n)
        for c in self.comp_classes:
            tp, fp, fn = match_components(pred == c, gt == c, self.thr)
            self.counts[c][0] += tp
            self.counts[c][1] += fp
            self.counts[c][2] += fn

    def result(self):
        inter = np.diag(self.conf).astype(float)
        union = self.conf.sum(0) + self.conf.sum(1) - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
        present = self.conf.sum(1) > 0
        out = {"mIoU": float(np.nanmean(iou[present]))}
        out.update({f"iou_{C16.names[c]}": float(iou[c]) for c in range(self.n)})
        for c, (tp, fp, fn) in self.counts.items():
            p = tp / (tp + fp) if tp + fp else float("nan")
            r = tp / (tp + fn) if tp + fn else float("nan")
            f = 2 * p * r / (p + r) if tp else 0.0
            name = C16.names[c]
            out.update({f"P_{name}": p, f"R_{name}": r, f"F1_{name}": f})
        return out


def match_components(pred_mask, gt_mask, thr=0.5):
    """(TP, FP, FN) of 4-connected components matched one-to-one (Hungarian on IoU >= thr)."""
    npred, pc = cv2.connectedComponents(pred_mask.astype(np.uint8), connectivity=4)
    ngt, gc = cv2.connectedComponents(gt_mask.astype(np.uint8), connectivity=4)
    npred, ngt = npred - 1, ngt - 1
    if npred == 0 or ngt == 0:
        return 0, npred, ngt
    both = (pc > 0) | (gc > 0)
    inter = np.bincount(gc[both] * (npred + 1) + pc[both], minlength=(ngt + 1) * (npred + 1))
    inter = inter.reshape(ngt + 1, npred + 1)[1:, 1:].astype(float)
    area_g = np.bincount(gc.ravel(), minlength=ngt + 1)[1:].astype(float)
    area_p = np.bincount(pc.ravel(), minlength=npred + 1)[1:].astype(float)
    iou = inter / (area_g[:, None] + area_p[None, :] - inter)
    rows, cols = linear_sum_assignment(-iou)
    tp = int((iou[rows, cols] >= thr).sum())
    return tp, npred - tp, ngt - tp


def fallback_check(res, cfg):
    """True if the parser is good enough (no fallback needed)."""
    return (res["iou_wall"] >= cfg["wall_iou"] and res["F1_door"] >= cfg["opening_f1"]
            and res["F1_window"] >= cfg["opening_f1"])
