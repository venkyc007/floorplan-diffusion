"""Novelty step 3 - train a large-hole FILL model on the large-omission set (parsed context -> true layout).

    python -m src.novelty.train_fill --config configs/novelty/fill_unet.yaml --train large_train_0.zip large_train_1.zip \
        --val large_tune_0.zip large_tune_1.zip --out <run dir> --init16 <B4 best.pt> [--no-ffc] [--smoke]

Same recipe as B4 (src/stage2/complete/train16.py: AdamW, warm-up, cosine over the time budget, grad clip, EMA,
fp16, the 14 constraint losses, last.pt + automatic resume), with three differences:
  * data: windows from src/novelty/data.FillSet (the parser's map as context, ground truth in the hole);
  * model: build_large (optional Fourier context blocks; --no-ffc switches them off for the ablation);
  * validation on fixed TUNING-drawing windows with real erased omissions:
      U-Net      one pass -> hole accuracy, hole mIoU, wall IoU in the hole
      diffusion  one denoising pass from the true plan noised to t = 800 (what the hybrid asks of it: finish a
                 heavily noised sketch) -> the same scores; best.pt by hole mIoU.
"""

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.novelty.data import FillSet, fixed_fill_windows, load_set
from src.novelty.models import build_large, warm_start
from src.stage2.complete.gates34 import TINY, hole_scores
from src.stage2.complete.train16 import append_csv, lr_factor
from src.stage2.data.classes16 import C16

WALL = C16.index["wall"]
VAL_T = 800


@torch.no_grad()
def validate(model, val, device, batch=16):
    cond, target, mask = val
    model.eval()
    preds = []
    g = torch.Generator(device=device).manual_seed(0)
    for i in range(0, len(cond), batch):
        c, x0 = cond[i:i + batch].to(device), target[i:i + batch].to(device) * 2 - 1
        if model.mcfg["kind"] == "unet":
            preds.append(model.predict(c).cpu())
        else:
            t = torch.full((len(c),), VAL_T, device=device, dtype=torch.long)
            noise = torch.randn(x0.shape, generator=g, device=device)
            x_t = model.train_scheduler.add_noise(x0, noise, t)
            preds.append(model(x_t, c, t).argmax(1).cpu())
    pred = torch.cat(preds)
    acc, miou = hole_scores(pred, target, mask)
    h = mask[:, 0] > 0
    p, t = (pred == WALL) & h, (target.argmax(1) == WALL) & h
    model.train()
    return {"hole_acc": acc, "hole_mIoU": miou, "wall_IoU": float((p & t).sum() / max(int((p | t).sum()), 1))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init16", default=None, help="B4 best.pt (warm start; ignored when resuming)")
    ap.add_argument("--no-ffc", action="store_true", help="ablation: no Fourier context blocks")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.no_ffc:
        cfg["model"]["ffc"] = {"enabled": False}
    tc, mcfg, dc = cfg["train"], cfg["model"], cfg["data"]
    if args.smoke:
        mcfg = cfg["model"] = mcfg | TINY
        tc = tc | {"batch_size": 2, "warmup_steps": 2, "log_every": 1}
        dc = dc | {"num_workers": 0, "window": 128, "val_windows": 4}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = tc["fp16"] and device.type == "cuda"
    torch.manual_seed(tc["seed"])
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    model = build_large(mcfg)
    last = out / "checkpoints" / "last.pt"
    if args.init16 and not last.exists() and not args.smoke:
        warm_start(model, args.init16)
        print("warm start from", args.init16, flush=True)
    model.to(device)
    ema = build_large(mcfg).to(device).eval()                  # never deep-copy a model with Fourier hooks
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step, best, hours_done = 0, -1.0, 0.0
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, best, hours_done = ck["step"], ck["best"], ck["hours"]
        torch.manual_seed(tc["seed"] + step)
        print(f"resumed at step {step} ({hours_done:.2f} h, best {best:.4f})", flush=True)

    samples = load_set(args.train, limit=40 if args.smoke else None)
    val = fixed_fill_windows(load_set(args.val, limit=40 if args.smoke else None), dc["val_windows"], win=dc["window"])[:3]
    loader = torch.utils.data.DataLoader(FillSet(samples, win=dc["window"], intact_frac=dc["intact_frac"]),
                                         batch_size=tc["batch_size"], num_workers=dc["num_workers"], drop_last=True,
                                         pin_memory=device.type == "cuda", persistent_workers=dc["num_workers"] > 0)
    budget, session = tc["time_limit_hours"], time.time()
    last_ckpt = last_val = time.time()

    def hours():
        return hours_done + (time.time() - session) / 3600

    def save(path):
        state = {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                 "scaler": scaler.state_dict(), "step": step, "best": best, "hours": hours(), "config": cfg}
        torch.save(state, str(path) + ".tmp")
        os.replace(str(path) + ".tmp", path)

    n_new = sum(p.numel() for n, p in model.named_parameters() if n.startswith("spectral."))
    print(f"{mcfg['kind']} | Fourier blocks: {n_new / 1e6:.2f} M new params | "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M params | {len(samples)} train versions | "
          f"{len(val[0])} val windows | device {device} | batch {tc['batch_size']}", flush=True)
    t_log, n_log, parts_acc = time.time(), 0, {}
    for cond, target, mask in loader:
        cond, target, mask = cond.to(device), target.to(device), mask.to(device)
        f = lr_factor(step, hours() / budget, tc["warmup_steps"])
        for gp in opt.param_groups:
            gp["lr"] = tc["lr"] * f
        with torch.autocast(device_type=device.type, enabled=amp):
            loss = model.loss(cond, target, mask, step=step)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        scaler.step(opt)
        scaler.update()
        with torch.no_grad():
            for e, p in zip(ema.parameters(), model.parameters()):
                e.mul_(tc["ema_decay"]).add_(p.detach(), alpha=1 - tc["ema_decay"])
            for e, b in zip(ema.buffers(), model.buffers()):
                e.copy_(b)
        step += 1
        n_log += len(cond)
        for k, v in model.last_parts.items():
            parts_acc[k] = parts_acc.get(k, 0.0) + float(v)
        if step % tc["log_every"] == 0:
            sps = n_log / (time.time() - t_log)
            parts = {k: round(v / tc["log_every"], 6) for k, v in parts_acc.items()}
            print(f"step {step}: {parts} lr x{f:.3f} {sps:.1f} samples/s {hours():.2f} h", flush=True)
            append_csv(out / "train_log.csv", {"step": step, "hours": round(hours(), 4), "lr_factor": round(f, 4),
                                               "samples_per_s": round(sps, 2), **parts})
            t_log, n_log, parts_acc = time.time(), 0, {}
        now = time.time()
        stop = hours() >= budget or (now - t_start) / 3600 >= budget - 0.03 or (args.smoke and step >= 4)
        if now - last_val >= tc["val_minutes"] * 60 or stop or (args.smoke and step % 2 == 0):
            res = validate(ema, val, device)
            print(f"VAL step {step}: " + " ".join(f"{k} {v:.4f}" for k, v in res.items()), flush=True)
            append_csv(out / "val_log.csv", {"step": step, "hours": round(hours(), 4), **{k: round(v, 6) for k, v in res.items()}})
            if res["hole_mIoU"] > best:
                best = res["hole_mIoU"]
                torch.save({"model": ema.state_dict(), "step": step, "config": cfg, "val": res},
                           out / "checkpoints" / "best.pt")
            last_val = time.time()
        if now - last_ckpt >= tc["checkpoint_minutes"] * 60 or stop or args.smoke:
            save(last)
            last_ckpt = time.time()
        if stop:
            break
    torch.save({"model": ema.state_dict(), "step": step, "config": cfg}, out / "checkpoints" / "ema_last.pt")
    print(f"stopped at step {step}, {hours():.2f} h, best hole mIoU {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
