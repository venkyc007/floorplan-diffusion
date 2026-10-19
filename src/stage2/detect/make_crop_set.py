"""Detector v3 data: drawing crops + the parser's map of the same crop + the omission target, at PARSER scale
(walls ~8 px) (user, 2026-10-19).

    python -m src.stage2.detect.make_crop_set --drawings <parser drawings> --parser <best.pt> --out crops.zip \
        [--ids ids.txt] [--minutes 20] [--seed 0] [--crops 4] [--size 384]

Versions per typed drawing exactly as make_parsed_set.py (intact + 2 single image-level omissions, erasure v2,
kinds balanced). Each version is parsed once on the GPU at scale 1. From each version `crops` crops of size x size:
omitted versions -> half centred on the omission (jittered by up to size / 4), half on random building pixels;
intact versions -> all on random building pixels. Saved per crop: <pid>_v<k>_c<j>_img.png (greyscale drawing),
_lab.png (parser's 16-class map), _tgt.png (MISSING index on the omission's pixels, else 0). Outside the drawing:
white paper / background / 0.
"""

import argparse
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

from src.stage2.data.classes16 import C16
from src.stage2.detect.make_parsed_set import Versions, identity, png
from src.stage2.detect.omissions import M
from src.stage2.parse.dataset import pad_to_multiple
from src.stage2.parse.train import Parser

BG = C16.index["background"]


def parse_full(model, img, device):
    pi, _ = pad_to_multiple(img, img, 32, 255, 255)
    x = torch.from_numpy(pi)[None, None].float().to(device) / 255
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        return model(x).argmax(1)[0, : img.shape[0], : img.shape[1]].cpu().numpy().astype(np.uint8)


def crop(a, cy, cx, size, fill):
    out = np.full((size, size), fill, a.dtype)
    y0, x0 = cy - size // 2, cx - size // 2
    h, w = a.shape
    sy0, sx0, sy1, sx1 = max(0, y0), max(0, x0), min(h, y0 + size), min(w, x0 + size)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[sy0:sy1, sx0:sx1]
    return out


def centres(lab, mask, n, size, rng):
    out = []
    build = np.argwhere(lab != BG)
    om = np.argwhere(mask) if mask is not None else np.zeros((0, 2), int)
    for j in range(n):
        if len(om) and j < n // 2:
            cy, cx = om[rng.integers(len(om))]
            cy, cx = cy + int(rng.integers(-size // 4, size // 4 + 1)), cx + int(rng.integers(-size // 4, size // 4 + 1))
        elif len(build):
            cy, cx = build[rng.integers(len(build))]
        else:
            cy, cx = lab.shape[0] // 2, lab.shape[1] // 2
        out.append((int(cy), int(cx)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drawings", required=True)
    ap.add_argument("--parser", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crops", type=int, default=4)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.parser, map_location=device, weights_only=False)
    model = Parser(ck["config"]["model"], encoder_weights=None).to(device).eval()
    model.load_state_dict(ck["model"])
    plans = sorted(p.name[:-8] for p in Path(args.drawings).glob("*_img.png"))
    if args.ids:
        keep = set(Path(args.ids).read_text(encoding="utf-8").split())
        plans = [p for p in plans if p in keep]
    plans = plans[: args.limit]
    loader = torch.utils.data.DataLoader(Versions(args.drawings, plans, args.seed), batch_size=None, shuffle=False,
                                         num_workers=args.workers, collate_fn=identity)
    rng = np.random.default_rng(args.seed + 99)
    n, counts = 0, {}
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as z:
        for pid, versions in loader:
            for k, (img, mask, typ) in enumerate(versions):
                lab = parse_full(model, img, device)
                tgt = np.zeros(img.shape, np.uint8)
                if mask is not None:
                    tgt[mask] = M[typ]
                    counts[typ] = counts.get(typ, 0) + 1
                for j, (cy, cx) in enumerate(centres(lab, mask, args.crops, args.size, rng)):
                    z.writestr(f"{pid}_v{k}_c{j}_img.png", png(crop(img, cy, cx, args.size, 255)))
                    z.writestr(f"{pid}_v{k}_c{j}_lab.png", png(crop(lab, cy, cx, args.size, BG)))
                    z.writestr(f"{pid}_v{k}_c{j}_tgt.png", png(crop(tgt, cy, cx, args.size, 0)))
            n += bool(versions)
            if n % 50 == 0:
                print(f"{n} drawings ({time.time() - t0:.0f} s) omissions {counts}", flush=True)
            if time.time() - t0 > args.minutes * 60:
                print("time budget reached", flush=True)
                break
    print(f"done: {n} drawings, omissions {counts}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
