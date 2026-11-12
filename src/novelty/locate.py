"""Novelty step 4 - the region LOCATOR: "is a large part of this plan missing, and where?"

    python -m src.novelty.locate --config configs/novelty/locate.yaml --train large_train_*.zip \
        --val large_tune_*.zip --out <run dir> [--smoke]

The shipped CHECK looks at small neighbourhoods, so a missing room only shows up as small symptoms (a door, a
sink) and the fill gets a small window; its one region rule (R8) fired on 0 of 20 large omissions. The locator
looks at the WHOLE parsed plan and marks the area where structure is missing:
  input   the parser's 16-class map of the drawing (one-hot, map scale, walls 3 px) - the map only, never the
          drawing, so it cannot learn erasure smudges, only what a plan with a missing part looks like
          (an empty oversized room, a facade stretch without windows, fixtures of two room types in one room)
  output  per-pixel probability "this belongs to a missing area" (the room + its walls / the erased rectangle)
  model   U-Net with an ImageNet ResNet-34 encoder (as detector v3), BCE + Dice
  data    src/novelty/data.LocateSet: 60% erased versions, 40% intact drawings (target all zero)
Validation (TUNING drawings, whole maps): pixel IoU on erased versions, and at the default flag rule (prob >= 0.5,
blob >= 150 px): share of omissions hit by a flag and flags per intact drawing. best.pt by IoU - false alarm rate.

At inference (also used by the pipeline, src/novelty/pipeline.py):
    prob = region_prob(model, parsed)                       # (H, W)
    flags = region_candidates(prob, th, min_area)           # Candidate("region", box, ["L"], score, mask=blob)
"""

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.novelty.data import BG, LocateSet, arrays, load_set
from src.stage2.constraints.rules import components
from src.stage2.data.classes16 import C16
from src.stage2.detect import rules as R


def build_locator(mcfg, weights="imagenet"):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=mcfg["encoder"], encoder_weights=weights, in_channels=C16.n, classes=1)


def load_locator(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_locator(ck["config"]["model"], None)
    model.load_state_dict(ck["model"])
    return model.eval().to(device)


def one_hot(lab):
    return F.one_hot(lab, C16.n).permute(0, 3, 1, 2).float()


@torch.no_grad()
def region_prob(models, parsed, device="cpu"):
    """Probability map (H, W) for one parsed map; `models`: one locator or a list (their mean)."""
    models = models if isinstance(models, (list, tuple)) else [models]
    H, W = parsed.shape
    ph, pw = -(-max(H, 64) // 32) * 32, -(-max(W, 64) // 32) * 32
    pad = np.full((ph, pw), BG, np.uint8)
    pad[:H, :W] = parsed
    x = one_hot(torch.from_numpy(pad.astype(np.int64))[None].to(device))
    p = sum(torch.sigmoid(m(x))[0, 0] for m in models) / len(models)
    return p[:H, :W].float().cpu().numpy()


def region_candidates(prob, th=0.5, min_area=150, pad=2):
    """Region flags: connected blobs of prob >= th with at least min_area px. Box = the blob's box + pad."""
    out = []
    for m in components(prob >= th):
        if m.sum() >= min_area:
            out.append(R.Candidate("region", R.box_of(m, m.shape, pad=pad), ["L"], 10.0 + float(prob[m].mean()), None, m))
    return out


def hit(flags, area, min_iou=0.0):
    """Is the omission area hit by a flag (box overlap; with min_iou: box IoU with the area's box at least that)?"""
    ys, xs = np.nonzero(area)
    a = (ys.min(), xs.min(), ys.max(), xs.max())
    for f in flags:
        b = f.box
        ih, iw = min(a[2], b[2]) - max(a[0], b[0]) + 1, min(a[3], b[3]) - max(a[1], b[1]) + 1
        if ih <= 0 or iw <= 0:
            continue
        inter = ih * iw
        union = (a[2] - a[0] + 1) * (a[3] - a[1] + 1) + (b[2] - b[0] + 1) * (b[3] - b[1] + 1) - inter
        if inter / union >= min_iou and area[b[0]:b[2] + 1, b[1]:b[3] + 1].any():
            return True
    return False


@torch.no_grad()
def validate(model, samples, device, th=0.5, min_area=150):
    model.eval()
    ious, hits, hits50, fa, n_intact, extra = [], [], [], 0, 0, 0
    for s in samples:
        parsed, _, om = arrays(s)
        prob = region_prob(model, parsed, device)
        flags = region_candidates(prob, th, min_area)
        area = om > 0
        if s["version"] == 0:
            n_intact += 1
            fa += len(flags)
        else:
            p = prob >= th
            ious.append((p & area).sum() / max((p | area).sum(), 1))
            hits.append(hit(flags, area))
            hits50.append(hit(flags, area, 0.5))
            extra += sum(1 for f in flags if not area[f.box[0]:f.box[2] + 1, f.box[1]:f.box[3] + 1].any())
    model.train()
    return {"pixel_IoU": float(np.mean(ious)), "found": float(np.mean(hits)), "found_box_iou50": float(np.mean(hits50)),
            "flags_per_intact": fa / max(n_intact, 1), "stray_flags_per_erased": extra / max(len(ious), 1)}


def dice_loss(logits, target):
    p = torch.sigmoid(logits.float())
    inter = (p * target).sum((1, 2, 3))
    return (1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)).mean()


def append_csv(path, row):
    import csv
    new = not Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tc, dc = cfg["train"], cfg["data"]
    if args.seed is not None:
        tc["seed"] = args.seed
    if args.smoke:
        tc = tc | {"batch_size": 2, "warmup_steps": 2, "log_every": 1}
        dc = dc | {"num_workers": 0, "canvas": 128}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    torch.manual_seed(tc["seed"])
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    samples = load_set(args.train, limit=40 if args.smoke else None)
    vsamples = load_set(args.val, limit=12 if args.smoke else None)
    loader = torch.utils.data.DataLoader(LocateSet(samples, dc["canvas"], dc["erased_frac"]), batch_size=tc["batch_size"],
                                         num_workers=dc["num_workers"], drop_last=True, pin_memory=amp,
                                         persistent_workers=dc["num_workers"] > 0)
    model = build_locator(cfg["model"], None if args.smoke else "imagenet").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step, best, budget = 0, -9.0, tc["time_limit_hours"]
    hours = lambda: (time.time() - t_start) / 3600
    print(f"locator: {len(samples)} train versions ({sum(s['version'] > 0 for s in samples)} erased), "
          f"{len(vsamples)} val versions, device {device}", flush=True)
    last_val, t_log, losses = time.time(), time.time(), []
    for lab, tgt in loader:
        x, y = one_hot(lab.to(device)), tgt.to(device)
        f = (step + 1) / tc["warmup_steps"] if step < tc["warmup_steps"] else 0.5 * (1 + math.cos(math.pi * min(hours() / budget, 1)))
        for g in opt.param_groups:
            g["lr"] = tc["lr"] * f
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits.float(), y) + dice_loss(logits, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        scaler.step(opt)
        scaler.update()
        step += 1
        losses.append(loss.item())
        if step % tc["log_every"] == 0:
            print(f"step {step}: loss {np.mean(losses):.4f} lr x{f:.3f} "
                  f"{tc['log_every'] * tc['batch_size'] / (time.time() - t_log):.1f} samples/s {hours():.2f} h", flush=True)
            append_csv(out / "train_log.csv", {"step": step, "hours": round(hours(), 4), "loss": round(float(np.mean(losses)), 5)})
            t_log, losses = time.time(), []
        stop = hours() >= budget or (args.smoke and step >= 4)
        if time.time() - last_val >= tc["val_minutes"] * 60 or stop or (args.smoke and step % 2 == 0):
            res = validate(model, vsamples, device)
            score = res["pixel_IoU"] - res["flags_per_intact"]
            print(f"VAL step {step}: " + " ".join(f"{k} {v:.3f}" for k, v in res.items()) + f" score {score:.3f}", flush=True)
            append_csv(out / "val_log.csv", {"step": step, "hours": round(hours(), 4), **{k: round(v, 5) for k, v in res.items()}})
            state = {"model": model.state_dict(), "step": step, "config": cfg, "val": res}
            if score > best:
                best = score
                torch.save(state, out / "checkpoints" / "best.pt")
            torch.save(state, str(out / "checkpoints" / "last.pt") + ".tmp")
            os.replace(str(out / "checkpoints" / "last.pt") + ".tmp", out / "checkpoints" / "last.pt")
            last_val = time.time()
        if stop:
            break
    print(f"stopped at step {step}, {hours():.2f} h, best score {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
