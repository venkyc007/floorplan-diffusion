"""Novelty step 1 - the "large omission set": what the pipeline REALLY sees when a large part of a drawing is missing.

    python -m src.novelty.build_set --drawings <folder of parser drawings> --parser <parser best.pt> \
        --out large_train.zip [--ids ids.txt] [--versions 3] [--minutes 50] [--seed 0] [--shard 0/2]

Why: the shipped large-hole models (B4) were trained on clean ground-truth label maps, but in the pipeline the hole
sits in the PARSER's map of the erased drawing (merged rooms, missing doors, a slightly different scale). On val
windows B4 fixed 29 / 30 large holes, on real drawings 6 / 20 (outputs/stage2/large_fill_compare_v2.md). This set
closes that gap: every sample is made exactly as the pipeline makes its input.

For each typed drawing (random order, seeded, so a time-limited run covers a random subset):
  v0        the intact drawing
  v1..vK    ONE large omission each, erased on the drawing itself with the evaluation's own generators
            (src/stage2/data/erasure_large.py: "room (whole)" 2-15% and "region" 5-25% of the building, kinds
            alternating, a kind the drawing lacks -> the other kind)
Each version goes through the pipeline's PARSE (src/stage2/pipeline/run.parse: parser v2B, the scale choice, the
rescale to 3 px walls). Saved per version (PNG, all at the parsed map's size):
  <pid>_v<k>_map.png   the parser's 16-class map of the (erased) drawing      = the context the fill will see
  <pid>_v<k>_gt.png    the ground-truth 16-class map, resized (nearest)        = what the fill should draw
  <pid>_v<k>_om.png    0 nothing, 1 omission area (the room + its walls / the rectangle inside the building),
                       2 erased structure (walls, doors, windows, fixtures): what "found" / "fixed" are scored on
and one line per version in meta.csv (pid, version, kind, size_pct, size_bin, map height, width).
CPU workers erase while the GPU parses; --shard i/n lets two processes (one per GPU) split the drawings.
"""

import argparse
import csv
import io
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

from src.stage2.detect.image_eval import LARGE_KINDS, omit
from src.stage2.parse.dataset import read_pair
from src.stage2.parse.train import Parser
from src.stage2.pipeline import run as pipeline


class Versions(torch.utils.data.Dataset):
    """One drawing -> [(image, structure mask, area mask, kind, meta)] (v0 intact, then K large omissions)."""

    def __init__(self, folder, plans, seed, versions):
        self.folder, self.plans, self.seed, self.versions = folder, plans, seed, versions

    def __len__(self):
        return len(self.plans)

    def __getitem__(self, i):
        pid = self.plans[i]
        img, lbl = read_pair(self.folder, pid)
        if (lbl == 255).any():                      # untyped plan: no room types, skipped everywhere in Stage 2
            return pid, None, []
        rng = np.random.default_rng([self.seed, sum(pid.encode())] + list(pid.encode()[-6:]))
        out = [(img, None, None, "intact", {})]
        first = int(rng.integers(2))
        for k in range(self.versions):
            for kind in (LARGE_KINDS[(first + k) % 2], LARGE_KINDS[(first + k + 1) % 2]):
                o = omit(kind, img, lbl, rng)
                if o is not None:
                    out.append((o[0], o[1], o[5], kind, o[4]))
                    break
        return pid, lbl, out


def identity(x):
    return x


def png(a):
    return cv2.imencode(".png", a)[1].tobytes()


def to_map(mask, shape):
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


INK_GREY = 128            # a drawing pixel darker than this is "ink"


def ink_map(img, shape):
    """Where the drawing has ANY ink, at map size: the share of ink pixels (grey < INK_GREY) under each map pixel,
    0-255. The drawing is binarised first, so paper tone, scan noise and the smudges an erasure can leave are all
    invisible in it; what remains is walls, symbols, furniture and text. A room that was never drawn has no label
    and no furniture, which the parser's 16-class map cannot show."""
    ink = (img < INK_GREY).astype(np.float32)
    return np.round(255 * cv2.resize(ink, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)).astype(np.uint8)


def load_parser(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = Parser(ck["config"]["model"], encoder_weights=None)
    model.load_state_dict(ck["model"])
    return model.eval().to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drawings", required=True)
    ap.add_argument("--parser", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default=None, help="text file of plan ids to keep (e.g. the tuning drawings)")
    ap.add_argument("--versions", type=int, default=3)
    ap.add_argument("--minutes", type=float, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ink", action="store_true", help="also store the ink map (_ink.png) and the grey drawing at "
                                                       "map size (_img.png; for trace checks only)")
    args = ap.parse_args()
    t0 = time.time()
    device = pipeline.DEVICE
    parser = load_parser(args.parser, device)
    pipeline.models = lambda: {"parser": parser}       # parse() only needs the parser (no detectors / fill models)
    pc = pipeline.cfg()["parse"]
    plans = sorted(p.name[:-8] for p in Path(args.drawings).glob("*_img.png"))
    if args.ids:
        keep = set(Path(args.ids).read_text(encoding="utf-8").split())
        plans = [p for p in plans if p in keep]
    plans = [plans[i] for i in np.random.default_rng(args.seed).permutation(len(plans))]
    si, sn = (int(v) for v in args.shard.split("/"))
    plans = plans[si::sn][: args.limit]
    loader = torch.utils.data.DataLoader(Versions(args.drawings, plans, args.seed, args.versions), batch_size=None,
                                         shuffle=False, num_workers=args.workers, collate_fn=identity)
    rows, n, counts = [], 0, {}
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as z:
        for pid, lbl, versions in loader:
            for k, (img, structure, area, kind, meta) in enumerate(versions):
                parsed, _, _ = pipeline.parse(img, pc)
                gt = cv2.resize(lbl, (parsed.shape[1], parsed.shape[0]), interpolation=cv2.INTER_NEAREST)
                om = np.zeros(parsed.shape, np.uint8)
                if structure is not None:
                    om[to_map(area, parsed.shape)] = 1
                    om[to_map(structure, parsed.shape)] = 2
                    if not (om == 2).any():
                        continue                   # the erased structure vanished at map scale: unusable case
                z.writestr(f"{pid}_v{k}_map.png", png(parsed))
                z.writestr(f"{pid}_v{k}_gt.png", png(gt))
                z.writestr(f"{pid}_v{k}_om.png", png(om))
                if args.ink:
                    z.writestr(f"{pid}_v{k}_ink.png", png(ink_map(img, parsed.shape)))
                    z.writestr(f"{pid}_v{k}_img.png", png(cv2.resize(img, (parsed.shape[1], parsed.shape[0]),
                                                                    interpolation=cv2.INTER_AREA)))
                rows.append({"pid": pid, "version": k, "kind": kind, "size_pct": meta.get("size_pct", ""),
                             "size_bin": meta.get("size_bin", ""), "h": parsed.shape[0], "w": parsed.shape[1]})
                counts[kind] = counts.get(kind, 0) + 1
            n += bool(versions)
            if n % 50 == 0 and versions:
                print(f"shard {args.shard}: {n} drawings ({time.time() - t0:.0f} s) {counts}", flush=True)
            if time.time() - t0 > args.minutes * 60:
                print("time budget reached", flush=True)
                break
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=["pid", "version", "kind", "size_pct", "size_bin", "h", "w"])
        w.writeheader()
        w.writerows(rows)
        z.writestr("meta.csv", buf.getvalue())
    print(f"done shard {args.shard}: {n} drawings, {counts}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
