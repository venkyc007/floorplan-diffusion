"""Detector v3 "direct": missing components from the DRAWING + the parse (user, 2026-10-19).

    python -m src.stage2.detect.train_detector_v3 --config configs/stage2/detector_v3.yaml --crops crops_train.zip \
        --val-crops crops_val.zip --out <run dir> [--v1 v1.pt --v2 v2.pt] [--smoke]

Input (17 channels, parser scale, walls ~8 px): the greyscale drawing crop (normalised as the parser's input) + the
16-class one-hot of parser v2B's map of the same crop. Output: per-pixel missing type (MISSING, as detectors v1/v2).
Model: smp U-Net, ResNet-34 encoder with ImageNet weights (smp adapts the first conv to 17 channels). Loss: class-
weighted CE (sqrt(f_max / f_c), cap 10, frequencies from 300 sampled training crops) + Dice, as v1 / v2.
Crops: make_crop_set.py (384 x 384), random 90-degree rotation + flip. Time-limited, last.pt every 30 min + resume,
best.pt by mean F1 on the validation crops (parsed TUNING drawings).
At the end (--v1 / --v2): v1, v2 and v3 on the same validation crops -> <out>/compare.md. v1 / v2 see the parse
shrunk to completion scale (x 3/8, nearest) with their usual features; their prediction is scaled back (nearest).
"""

import argparse
import math
import os
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.stage2.data.classes16 import C16
from src.stage2.detect.omissions import MISSING
from src.stage2.detect.train_detector import N_OUT, DetMetrics, append_csv, build_model, features, rule_maps
from src.stage2.parse.losses import ParserLoss, class_weights

GREY_MEAN, GREY_STD = 0.449, 0.226


def load_crops(path):
    z = zipfile.ZipFile(path)
    names = sorted(n[:-8] for n in z.namelist() if n.endswith("_img.png"))
    return [(z.read(f"{n}_img.png"), z.read(f"{n}_lab.png"), z.read(f"{n}_tgt.png"), "_v0_" not in n) for n in names]


def dec(b):
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)


class CropSet(torch.utils.data.Dataset):
    def __init__(self, crops, augment=True):
        self.crops, self.augment, self.rng = crops, augment, None

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, i):
        img, lab, tgt = (dec(b) for b in self.crops[i][:3])
        if self.augment:
            if self.rng is None:
                self.rng = np.random.default_rng(torch.initial_seed() % 2 ** 32)
            k = int(self.rng.integers(4))
            flip = self.rng.random() < 0.5
            img, lab, tgt = (np.rot90(a[:, ::-1] if flip else a, k) for a in (img, lab, tgt))
        return (torch.from_numpy(np.ascontiguousarray(img)).float()[None] / 255,
                torch.from_numpy(np.ascontiguousarray(lab).astype(np.int64)),
                torch.from_numpy(np.ascontiguousarray(tgt).astype(np.int64)))


def inputs(img, lab):
    oh = F.one_hot(lab, C16.n).permute(0, 3, 1, 2).float()
    return torch.cat([(img - GREY_MEAN) / GREY_STD, oh], 1)


def build_v3(mcfg, weights="imagenet"):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=mcfg["encoder"], encoder_weights=weights, in_channels=1 + C16.n, classes=N_OUT)


@torch.no_grad()
def validate_v3(model, loader, device):
    model.eval()
    m = DetMetrics()
    for img, lab, tgt in loader:
        pred = model(inputs(img.to(device), lab.to(device))).argmax(1).cpu().numpy()
        for p, t in zip(pred, tgt.numpy()):
            m.update(p, t)
    model.train()
    return m.result()


@torch.no_grad()
def validate_old(model, loader, device, K):
    """v1 / v2 on the parse shrunk to completion scale; prediction scaled back to the crop."""
    model.eval()
    m = DetMetrics()
    for img, lab, tgt in loader:
        size = int(round(lab.shape[-1] * 3 / 8))
        pad = -(-size // 32) * 32                         # smp needs multiples of 32: background padding
        small = np.full((len(lab), pad, pad), C16.index["background"], np.uint8)
        for i, l in enumerate(lab):
            small[i, :size, :size] = cv2.resize(l.numpy().astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
        plan = torch.from_numpy(small.astype(np.int64))
        rules = torch.from_numpy(np.stack([rule_maps(s) for s in small]))
        pred = model(features(plan.to(device), rules, K)).argmax(1)[:, :size, :size].cpu().numpy().astype(np.uint8)
        for p, t in zip(pred, tgt.numpy()):
            m.update(cv2.resize(p, (t.shape[1], t.shape[0]), interpolation=cv2.INTER_NEAREST), t)
    return m.result()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--crops", required=True)
    ap.add_argument("--val-crops", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--v1", default=None)
    ap.add_argument("--v2", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tc = cfg["train"]
    if args.smoke:
        tc = tc | {"batch_size": 2, "workers": 0, "warmup_steps": 2, "log_every": 1}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    torch.manual_seed(tc["seed"])
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    crops, vcrops = load_crops(args.crops), load_crops(args.val_crops)
    vcrops = vcrops[: (4 if args.smoke else cfg["val"]["crops"])]
    ds = CropSet(crops)
    sampler = torch.utils.data.RandomSampler(ds, replacement=True, num_samples=10 ** 8)
    loader = torch.utils.data.DataLoader(ds, batch_size=tc["batch_size"], sampler=sampler, num_workers=tc["workers"],
                                         persistent_workers=tc["workers"] > 0, drop_last=True,
                                         pin_memory=device.type == "cuda")
    vloader = torch.utils.data.DataLoader(CropSet(vcrops, augment=False), batch_size=tc["batch_size"],
                                          num_workers=tc["workers"])
    rng = np.random.default_rng(777)
    counts = np.zeros(N_OUT)
    for i in rng.integers(len(crops), size=min(300, len(crops))):
        counts += np.bincount(dec(crops[i][2]).ravel(), minlength=N_OUT)
    crit = ParserLoss(class_weights(counts / counts.sum(), cfg["loss"]["ce_weight_cap"]), cfg["loss"]["dice"]).to(device)
    model = build_v3(cfg["model"], None if args.smoke else "imagenet").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step, best, hours_done = 0, -1.0, 0.0
    last = out / "checkpoints" / "last.pt"
    if last.exists():
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        step, best, hours_done = st["step"], st["best"], st["hours"]
    budget, session = tc["time_limit_hours"], time.time()

    def hours():
        return hours_done + (time.time() - session) / 3600

    def save(path, tag):
        st = {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step,
              "best": best, "hours": hours(), "config": cfg, "tag": tag, "class_weights": crit.weights.cpu().tolist()}
        torch.save(st, str(path) + ".tmp")
        os.replace(str(path) + ".tmp", path)

    print(f"detector v3: {len(crops)} train crops, {len(vcrops)} val crops, class weights "
          f"{[round(float(w), 2) for w in crit.weights]}", flush=True)
    last_ckpt = last_val = time.time()
    t_log, n_log, losses = time.time(), 0, []
    for img, lab, tgt in loader:
        x = inputs(img.to(device, non_blocking=True), lab.to(device, non_blocking=True))
        y = tgt.to(device, non_blocking=True)
        f = (step + 1) / tc["warmup_steps"] if step < tc["warmup_steps"] else \
            0.5 * (1 + math.cos(math.pi * min(hours() / budget, 1)))
        for g in opt.param_groups:
            g["lr"] = tc["lr"] * f
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
        loss, parts = crit(logits, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        scaler.step(opt)
        scaler.update()
        step += 1
        losses.append(loss.item())
        n_log += len(x)
        if step % tc["log_every"] == 0:
            sps = n_log / (time.time() - t_log)
            print(f"step {step}: loss {np.mean(losses):.4f} {sps:.1f} samples/s {hours():.2f} h", flush=True)
            append_csv(out / "train_log.csv", {"step": step, "hours": round(hours(), 4),
                                               "loss": round(float(np.mean(losses)), 5), "samples_per_s": round(sps, 2)})
            t_log, n_log, losses = time.time(), 0, []
        now = time.time()
        stop = hours() >= budget or (now - t_start) / 3600 >= budget - 0.05 or (args.smoke and step >= 4)
        if now - last_val >= tc["val_minutes"] * 60 or stop or (args.smoke and step % 2 == 0):
            res = validate_v3(model, vloader, device)
            print(f"VAL step {step}: mean F1 {res['mean_F1']:.3f} false alarms {res['false_alarm_rate']:.3f}", flush=True)
            append_csv(out / "val_log.csv", {"step": step, "hours": round(hours(), 4),
                                             **{k: round(float(v), 5) for k, v in res.items()}})
            if res["mean_F1"] > best:
                best = res["mean_F1"]
                save(out / "checkpoints" / "best.pt", "best")
            last_val = time.time()
        if now - last_ckpt >= tc["checkpoint_minutes"] * 60 or stop or args.smoke:
            save(last, "last")
            last_ckpt = time.time()
        if stop:
            break
    print(f"stopped at step {step}, {hours():.2f} h, best mean F1 {best:.4f}", flush=True)
    if args.v1 and args.v2:
        st = torch.load(out / "checkpoints" / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        rows = {"v3 (drawing + parse)": validate_v3(model, vloader, device)}
        for name, p in (("v1 (label-level)", args.v1), ("v2 (parsed maps)", args.v2)):
            ck = torch.load(p, map_location=device, weights_only=False)
            old = build_model(ck["config"]["model"]).to(device)
            old.load_state_dict(ck["model"])
            rows[name] = validate_old(old, vloader, device, ck["config"]["model"]["K"])
        names = MISSING[1:]
        L = ["# Detectors v1 / v2 / v3 on the same validation crops (parsed TUNING drawings, 384 px, parser scale)", "",
             f"{len(vcrops)} crops. Detected = >= 30% of an omitted component's pixels predicted as its type; false "
             "alarms = intact crops with >= 20 predicted missing pixels. v1 / v2 see the parse at completion scale.", "",
             "| detector | mean F1 | false alarms (intact crops) | " + " | ".join(f"F1 / detected {n}" for n in names) + " |",
             "|---|---|---|" + "---|" * len(names)]
        for name, r in rows.items():
            L.append(f"| {name} | {r['mean_F1']:.3f} | {r['false_alarm_rate']:.2f} | " +
                     " | ".join(f"{r['F1_' + n]:.2f} / {r['detected_' + n]:.2f}" for n in names) + " |")
        (out / "compare.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
