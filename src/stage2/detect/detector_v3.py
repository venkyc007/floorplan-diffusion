"""Detector v3 inference on a whole drawing (train_detector_v3.py): drawing + parse -> missing-type probabilities,
returned at COMPLETION scale in the same format as detection_eval.detector_probs (N_OUT, h, w), so the same
proposals() thresholds apply.

    probs = v3_probs(model, image, label)       # label: the pipeline's completion-scale parse (walls 3 px)

The parser-scale grid is rebuilt from the completion map: size x 8/3 (walls 3 -> 8 px); the drawing is resized to
that grid and the map upsampled (nearest) to it. Prediction in tiles of 1024 px (64 px overlap) when large; the
softmax is shrunk back to the completion map's size (area interpolation).
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.stage2.data.classes16 import C16
from src.stage2.detect.train_detector_v3 import build_v3, inputs


def load_v3(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build_v3(ck["config"]["model"], weights=None)
    m.load_state_dict(ck["model"])
    return m.eval()


@torch.no_grad()
def v3_probs(model, image, label, tile=1024, overlap=64):
    h, w = label.shape
    H, W = int(round(h * 8 / 3)), int(round(w * 8 / 3))
    img = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA if image.shape[0] > H else cv2.INTER_LINEAR)
    lab = cv2.resize(label, (W, H), interpolation=cv2.INTER_NEAREST)
    Hp, Wp = -(-H // 32) * 32, -(-W // 32) * 32
    pi = np.full((Hp, Wp), 255, np.uint8)
    pl = np.full((Hp, Wp), C16.index["background"], np.uint8)
    pi[:H, :W], pl[:H, :W] = img, lab
    dev = next(model.parameters()).device
    x = inputs(torch.from_numpy(pi)[None, None].float().to(dev) / 255, torch.from_numpy(pl.astype(np.int64))[None].to(dev))
    if Hp <= tile and Wp <= tile:
        p = F.softmax(model(x).float(), 1)[0]
    else:
        p = torch.zeros((model.segmentation_head[0].out_channels, Hp, Wp), device=dev)
        step = tile - 2 * overlap
        for y in range(0, Hp, step):
            for x0 in range(0, Wp, step):
                ya, xa = max(0, min(y - overlap, Hp - tile)), max(0, min(x0 - overlap, Wp - tile))
                t = x[..., ya:ya + min(tile, Hp), xa:xa + min(tile, Wp)]
                q = F.softmax(model(t).float(), 1)[0]
                yb, xb = min(y + step, Hp), min(x0 + step, Wp)
                p[:, y:yb, x0:xb] = q[:, y - ya:yb - ya, x0 - xa:xb - xa]
    p = p[:, :H, :W].cpu().numpy()
    return np.stack([cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA) for c in p])
