"""Train a 16-class FILL model (K1: constrained diffusion vs control) on completion-set windows.

    python -m src.stage2.complete.train16 --config configs/stage2/train16_constrained.yaml \
        --data <folder with completion_train.zip, completion_val.zip> --out <run dir> \
        --init <Stage 1 best.pt> [--smoke]

  * start: Stage 1 checkpoint, class-expanded (warm_start.py); ignored when resuming;
  * one process, one GPU (K1 runs the constrained and the control model side by side,
    one per T4); fp16 autocast for the network, constraint losses in fp32;
  * AdamW, warm-up then cosine over max_steps (the SAME for the constrained and control run,
    so the ablation is fair; time_limit_hours is only a safety stop), grad clip 1.0,
    EMA 0.999 (validation and best.pt use the EMA weights);
  * last.pt every checkpoint_minutes and at the end; automatic resume from <out>/checkpoints/last.pt;
  * every val_minutes: 256 fixed val windows, 20-step stochastic DDIM -> hole pixel accuracy,
    hole mIoU and the mean violation (in M) of every constraint on the samples -> val_log.csv;
    best.pt by hole mIoU.
"""

import argparse
import copy
import csv
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.stage2.complete.data16 import WindowSet, fixed_windows, load_plans
from src.stage2.complete.gates34 import TINY, hole_scores
from src.stage2.complete.models16 import build16
from src.stage2.complete.warm_start import warm_start
from src.stage2.complete.windows import model_inputs
from src.stage2.constraints import ops
from src.stage2.constraints.constraints import REGISTRY, Ctx, load_constants, one_hot, violation_map


def append_csv(path, row):
    new = not Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def lr_factor(step, progress, warmup):
    return (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def validate(model, val, device, steps, batch=32):
    """Hole accuracy / mIoU and mean constraint violation of DDIM samples on fixed val windows."""
    labels, masks = val
    model.eval()
    preds = []
    g = torch.Generator(device=device).manual_seed(0)
    for i in range(0, len(labels), batch):
        cond, _, _ = model_inputs(labels[i:i + batch], masks[i:i + batch])
        preds.append(model.predict(cond.to(device), steps=steps, generator=g).cpu())
    pred = torch.cat(preds)
    cond, target, mask = model_inputs(labels, masks)
    acc, miou = hole_scores(pred, target, mask)
    res = {"hole_acc": acc, "hole_mIoU": miou}
    P, M = one_hot(pred), ops.dil(mask, 7)
    ctx = Ctx(hole=mask, Y=target, known=target * (1 - mask), const=load_constants(), K=96)
    for name in REGISTRY:
        if name == "C17":
            continue
        v = violation_map(name, P, ctx if name not in ("C1", "C8") else Ctx(hole=mask, K=96))
        res[f"V_{name}"] = float(ops.masked_mean(v, M).mean())
    model.train()
    return res


class EMA:
    def __init__(self, model, decay):
        self.model, self.decay = copy.deepcopy(model).eval(), decay
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None, help="Stage 1 best.pt (class-expanded warm start)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tc, mcfg = cfg["train"], cfg["model"]
    if args.smoke:
        mcfg = mcfg | TINY
        tc = tc | {"batch_size": 2, "warmup_steps": 2, "log_every": 1}
        cfg["data"]["num_workers"] = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = tc["fp16"] and device.type == "cuda"
    torch.manual_seed(tc["seed"])
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    last = out / "checkpoints" / "last.pt"
    if args.init and not last.exists() and not args.smoke:
        model, _ = warm_start(mcfg["kind"], args.init, mcfg)
    else:
        model = build16(mcfg)
    model.to(device)
    ema = EMA(model, tc["ema_decay"])
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step, best, hours_done = 0, -1.0, 0.0
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.model.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, best, hours_done = ck["step"], ck["best"], ck["hours"]
        torch.manual_seed(tc["seed"] + step)
        print(f"resumed at step {step} ({hours_done:.2f} h, best hole mIoU {best:.4f})", flush=True)

    data = Path(args.data)
    plans = load_plans(data / cfg["data"]["completion_zip"], limit=20 if args.smoke else None, seed=0)
    vplans = load_plans(data / cfg["data"]["val_zip"])
    val = fixed_windows(vplans, 8 if args.smoke else 256, seed=1234)[:2]
    loader = torch.utils.data.DataLoader(WindowSet(plans), batch_size=tc["batch_size"],
                                         num_workers=cfg["data"]["num_workers"], drop_last=True,
                                         pin_memory=device.type == "cuda",
                                         persistent_workers=cfg["data"]["num_workers"] > 0)
    budget = tc["time_limit_hours"]
    session = time.time()
    last_ckpt = last_val = time.time()
    val_steps = 5 if args.smoke else 20

    def hours():
        return hours_done + (time.time() - session) / 3600

    def save(path):
        state = {"model": model.state_dict(), "ema": ema.model.state_dict(), "opt": opt.state_dict(),
                 "scaler": scaler.state_dict(), "step": step, "best": best, "hours": hours(), "config": cfg}
        torch.save(state, str(path) + ".tmp")
        os.replace(str(path) + ".tmp", path)

    print(f"{mcfg['kind']} constraints={'ON' if mcfg['constraints']['enabled'] else 'OFF'} | "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M params | {len(plans)} plans | "
          f"device {device} | batch {tc['batch_size']}", flush=True)
    t_log, n_log, parts_acc = time.time(), 0, {}
    for cond, target, mask in loader:
        cond, target, mask = cond.to(device), target.to(device), mask.to(device)
        progress = step / tc["max_steps"] if tc.get("max_steps") else hours() / budget
        f = lr_factor(step, progress, tc["warmup_steps"])
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
        ema.update(model)
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
        stop = hours() >= budget or (now - t_start) / 3600 >= budget - 0.05 or (args.smoke and step >= 4) \
            or (bool(tc.get("max_steps")) and step >= tc["max_steps"])
        if now - last_val >= tc["val_minutes"] * 60 or stop or (args.smoke and step % 2 == 0):
            t0 = time.time()
            res = validate(ema.model, val, device, val_steps)
            print(f"VAL step {step}: hole acc {res['hole_acc']:.4f} mIoU {res['hole_mIoU']:.4f} "
                  f"({time.time() - t0:.0f} s) " + " ".join(f"{k}={v:.4f}" for k, v in res.items() if k.startswith("V_")),
                  flush=True)
            append_csv(out / "val_log.csv", {"step": step, "hours": round(hours(), 4), **{k: round(v, 6) for k, v in res.items()}})
            if res["hole_mIoU"] > best:
                best = res["hole_mIoU"]
                torch.save({"model": ema.model.state_dict(), "step": step, "config": cfg, "val": res},
                           out / "checkpoints" / "best.pt")
            last_val = time.time()
        if now - last_ckpt >= tc["checkpoint_minutes"] * 60 or stop or args.smoke:
            save(last)
            last_ckpt = time.time()
        if stop:
            break
    print(f"stopped at step {step}, {hours():.2f} h, best hole mIoU {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
