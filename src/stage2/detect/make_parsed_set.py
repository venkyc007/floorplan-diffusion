"""Detector v2 data: what the detector really sees = the PARSER's map of a drawing with image-level omissions
(user, 2026-10-19).

    python -m src.stage2.detect.make_parsed_set --drawings <folder of parser drawings> --parser <best.pt> \
        --out parsed_train.zip [--ids ids.txt] [--minutes 40] [--seed 0]

For each typed drawing (file order, optionally only the plan ids in --ids), 3 versions:
  v0 intact, v1 and v2 with ONE image-level omission each (erasure v2, image_eval.omit), kinds balanced over
  door (a), door (b), window, wall, fixture, room, railing (each drawing continues the cycle; a kind the drawing
  lacks -> the next kind).
Each version is parsed with the given parser on the GPU at scale 1 (the parser set's walls are already ~8 px) and
the map is rescaled so walls = 3 px, exactly as the pipeline's parse() does (src/stage2/pipeline/run.py; the
multi-scale retry is not needed for these drawings).
Saved per version: <pid>_v<k>_map.png (16-class completion-scale parse) and <pid>_v<k>_tgt.png (MISSING index
of the omission's pixels, resized nearest; 0 elsewhere; all 0 for v0). Stops after --minutes (time budget);
CPU workers prepare the erased drawings while the GPU parses.
"""

import argparse
import io
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

from src.stage2.detect.image_eval import KINDS, omit
from src.stage2.detect.omissions import M
from src.stage2.parse.dataset import pad_to_multiple, read_pair
from src.stage2.parse.train import Parser
from src.stage2.pipeline.run import wall_thickness

WALL_PX, COMP_PX = 8, 3


class Versions(torch.utils.data.Dataset):
    def __init__(self, folder, plans, seed):
        self.folder, self.plans, self.seed = folder, plans, seed

    def __len__(self):
        return len(self.plans)

    def __getitem__(self, i):
        pid = self.plans[i]
        img, lbl = read_pair(self.folder, pid)
        if (lbl == 255).any():
            return pid, []
        rng = np.random.default_rng(self.seed + i)
        out = [(img, None, None)]
        for k in (2 * i, 2 * i + 1):
            for step in range(len(KINDS)):
                o = omit(KINDS[(k + step) % len(KINDS)], img, lbl, rng)
                if o is not None:
                    out.append((o[0], o[1], o[2]))
                    break
        return pid, out


def identity(x):
    return x


def parse_gpu(model, img, device):
    pi, _ = pad_to_multiple(img, img, 32, 255, 255)
    x = torch.from_numpy(pi)[None, None].float().to(device) / 255
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        lab = model(x).argmax(1)[0, : img.shape[0], : img.shape[1]].cpu().numpy().astype(np.uint8)
    t = wall_thickness(lab)
    f = COMP_PX / t if np.isfinite(t) and t > 0 else COMP_PX / WALL_PX
    size = (max(8, int(round(lab.shape[1] * f))), max(8, int(round(lab.shape[0] * f))))
    return cv2.resize(lab, size, interpolation=cv2.INTER_NEAREST)


def png(a):
    ok, buf = cv2.imencode(".png", a)
    return buf.tobytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drawings", required=True)
    ap.add_argument("--parser", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--seed", type=int, default=0)
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
    counts, n = {}, 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as z:
        for pid, versions in loader:
            for k, (img, mask, typ) in enumerate(versions):
                comp = parse_gpu(model, img, device)
                tgt = np.zeros(comp.shape, np.uint8)
                if mask is not None:
                    m = cv2.resize(mask.astype(np.uint8), (comp.shape[1], comp.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                    tgt[m] = M[typ]
                    counts[typ] = counts.get(typ, 0) + 1
                z.writestr(f"{pid}_v{k}_map.png", png(comp))
                z.writestr(f"{pid}_v{k}_tgt.png", png(tgt))
            n += bool(versions)
            if n % 50 == 0:
                print(f"{n} drawings ({time.time() - t0:.0f} s) omissions {counts}", flush=True)
            if time.time() - t0 > args.minutes * 60:
                print("time budget reached", flush=True)
                break
    print(f"done: {n} drawings, omissions {counts}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
