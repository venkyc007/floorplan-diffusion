"""Train the diffusion model or the U-Net baseline.

    python -m src.train --config configs/train_diffusion.yaml --out outputs/runs/diffusion
    python -m src.train --config configs/train_unet.yaml --out outputs/runs/unet
    python -m src.train --config configs/train_diffusion.yaml --out outputs/runs/smoke --smoke

On several GPUs (one process per GPU, see src/distributed.py):
    torchrun --standalone --nproc_per_node=2 -m src.train --config ... --out ...
`batch_size` in the config is PER GPU, so 2 GPUs see 2x the samples per step.

Built for Kaggle's free GPUs, whose sessions end after a time limit:
  - saves a checkpoint (OUT/checkpoints/last.pt) every `checkpoint_minutes`,
  - stops cleanly (and saves) once `time_limit_hours` have passed,
  - RESUMES automatically from OUT/checkpoints/last.pt when it exists.
So you can just run the same command again in a new session. Keep max_steps
the same across sessions: the learning-rate schedule is computed from it.

Every `val_every` steps since the last validation, or after `val_minutes`
minutes if that comes sooner (either one resets both),
it scores the model on a fixed set of val-benchmark cases (same cases every
time; diffusion uses `val_ddim_steps` sampling steps) and writes:
  OUT/val_log.csv                 step, strict/tolerant mIoU, door/window/room F1
  OUT/checkpoints/best.pt         weights with the best tolerant mIoU so far
  OUT/samples/step_XXXXXXX.png    input | prediction | ground truth
Training loss (averaged over GPUs) and speed go to OUT/train_log.csv.

--smoke uses the tiny settings in the config's `smoke:` section (CPU test).
"""

import argparse
import copy
import csv
import math
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from src.data.corruptions import to_model_input
from src.data.dataset import CorruptionDataset
from src.distributed import Distributed
from src.eval.figures import colorize
from src.eval.metrics import load_benchmark, sample_stats, summarize
from src.models import build_model
from src.models.backbone import count_parameters
from src.utils import load_config, project_path

VAL_SEED = 1234   # picks the fixed validation subset


# ------------------------------------------------------------- setup

def load_settings(path, smoke):
    """Read the YAML config; with --smoke, apply its `smoke:` overrides."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if smoke:
        for section, values in cfg["smoke"].items():
            cfg[section].update(values)
    return cfg


def lr_factor(step, warmup, total):
    """Linear warm-up to the full learning rate, then cosine decay to 0."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1 + math.cos(math.pi * progress))


def infinite(loader, sampler=None):
    """Loop over the DataLoader forever (training is counted in steps).
    With several GPUs, the sampler reshuffles each process's share every pass."""
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


class LossModule(nn.Module):
    """Computes the training loss inside forward().

    DistributedDataParallel only synchronises gradients for work done in
    forward(), and our models compute their loss in .loss(); this wrapper
    makes the loss part of forward()."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, cond, target, mask):
        return self.model.loss(cond, target, mask)


class EMA:
    """Exponential moving average of the weights: a smoothed copy of the
    model that usually samples better than the raw weights (diffusion)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)


def train_step(net, model, ema, opt, sched, scaler, batch, device, use_fp16, grad_clip):
    """One optimisation step (also used by src/speed_test.py). Returns the loss.

    net: LossModule(model), possibly wrapped in DistributedDataParallel;
    model: the plain model (for clipping and EMA)."""
    cond, target, mask = (t.to(device, non_blocking=True) for t in batch)
    with torch.autocast(device.type, dtype=torch.float16, enabled=use_fp16):
        loss = net(cond, target, mask)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)                       # so clipping sees the real gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(opt)
    scaler.update()
    sched.step()
    if ema:
        ema.update(model)
    return loss.item()


# ------------------------------------------------------------ validation

def val_subset(cfg):
    """The fixed validation cases: a seeded random choice from the val benchmark."""
    bench, meta = load_benchmark(project_path(cfg["data"]["val_benchmark"]))
    n = min(cfg["data"]["val_cases"], len(meta))
    idx = np.sort(np.random.default_rng(VAL_SEED).choice(len(meta), n, replace=False))
    cond = np.stack([to_model_input(bench["inputs"][i], bench["masks"][i]) for i in idx])
    return {"cond": torch.from_numpy(cond), "targets": bench["targets"][idx],
            "masks": bench["masks"][idx], "inputs": bench["inputs"][idx]}


@torch.no_grad()
def validate(model, val, device, steps=None, batch_size=32):
    """Score the model on the fixed val cases. Returns (metrics dict, predictions).
    steps: DDIM sampling steps (diffusion only; None = the config's ddim_steps)."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(0)  # same noise every validation
    preds = []
    for s in range(0, len(val["cond"]), batch_size):
        cond = val["cond"][s:s + batch_size].to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            preds.append(model.predict(cond, steps=steps, generator=gen).cpu().numpy())
    preds = np.concatenate(preds)
    scores = summarize([sample_stats(p, t, m) for p, t, m in zip(preds, val["targets"], val["masks"])])
    model.train()
    keep = ["mIoU", "tmIoU", "F1_door", "F1_window", "F1_room"]
    return {k: round(float(scores[k]), 4) for k in keep}, preds


def save_grid(val, preds, path, n):
    fig, axes = plt.subplots(n, 3, figsize=(7, 2.4 * n))
    for r in range(n):
        pred = np.where(val["masks"][r], preds[r], val["targets"][r]).astype(np.uint8)
        for c, (img, title) in enumerate([(val["inputs"][r], "input"), (pred, "prediction"),
                                          (val["targets"][r], "ground truth")]):
            axes[r, c].imshow(colorize(img), interpolation="nearest")
            axes[r, c].set_title(title, fontsize=9)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def append_csv(path, row):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


# ----------------------------------------------------------- checkpoints

def save_checkpoint(path, state):
    """Write to a temporary file first, then rename: a session killed mid-save
    can never leave a half-written last.pt behind."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def training_state(step, model, ema, opt, sched, scaler, best, cfg, hours):
    return {"step": step, "hours": hours, "model": model.state_dict(),
            "ema": ema.model.state_dict() if ema else None,
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(), "best_tmIoU": best, "config": cfg,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="run folder (checkpoints, logs, samples)")
    parser.add_argument("--smoke", action="store_true", help="tiny model, few steps (CPU test)")
    parser.add_argument("--max-steps", type=int, help="override train.max_steps")
    parser.add_argument("--time-limit-hours", type=float, help="override train.time_limit_hours")
    args = parser.parse_args()

    start = time.time()
    dist = Distributed()
    device, main_proc = dist.device, dist.is_main
    log = print if main_proc else (lambda *a, **k: None)   # only rank 0 prints
    cfg = load_settings(args.config, args.smoke)
    tc = cfg["train"]
    if args.max_steps:
        tc["max_steps"] = args.max_steps
    if args.time_limit_hours:
        tc["time_limit_hours"] = args.time_limit_hours
    out = Path(args.out)
    if main_proc:
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
        (out / "samples").mkdir(exist_ok=True)
    use_fp16 = tc["fp16"] and device.type == "cuda"
    # Different seed per process: different random corruptions on each GPU.
    # (DDP copies rank 0's initial weights to all processes.)
    torch.manual_seed(tc["seed"] + dist.rank)
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"process {dist.rank}/{dist.world_size} on {device} ({name})", flush=True)

    # Model, EMA (diffusion, main process only), optimiser, schedule, fp16 scaler.
    model = build_model(cfg["model"]).to(device)
    ema = EMA(model, tc["ema_decay"]) if cfg["model"]["kind"] == "diffusion" and main_proc else None
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, tc["warmup_steps"], tc["max_steps"]))
    scaler = torch.amp.GradScaler(device.type, enabled=use_fp16)
    step, best, hours_before = 0, -1.0, 0.0   # hours_before: training time of earlier sessions

    last = out / "checkpoints" / "last.pt"
    if last.exists():   # every process loads the same checkpoint
        ckpt = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if ema:
            ema.model.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        step, best, hours_before = ckpt["step"], ckpt["best_tmIoU"], ckpt.get("hours", 0.0)
        if dist.enabled:   # the saved random state is rank 0's: give each process a fresh one
            torch.manual_seed(tc["seed"] + dist.rank + step)
        else:
            torch.set_rng_state(ckpt["torch_rng"])
            if ckpt["cuda_rng"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        log(f"Resumed from {last} at step {step} (best tolerant mIoU {best:.4f})")
    elif main_proc:
        with open(out / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
    log(f"{cfg['model']['kind']}: {count_parameters(model) / 1e6:.2f}M parameters, "
        f"{dist.world_size} process(es) on {device.type}{' (fp16)' if use_fp16 else ''}, "
        f"batch {tc['batch_size']} per process = {tc['batch_size'] * dist.world_size} per step")

    net = LossModule(model)
    if dist.enabled:
        net = DistributedDataParallel(net, device_ids=[dist.local_rank] if device.type == "cuda" else None)

    # Data: a fresh random corruption per sample; each process gets its own share.
    dataset = CorruptionDataset(project_path(cfg["data"]["train_npz"]), load_config("corruption"))
    sampler = (DistributedSampler(dataset, num_replicas=dist.world_size, rank=dist.rank,
                                  shuffle=True, seed=tc["seed"], drop_last=True) if dist.enabled else None)
    loader = DataLoader(dataset, batch_size=tc["batch_size"], shuffle=sampler is None, sampler=sampler,
                        drop_last=True, num_workers=cfg["data"]["num_workers"],
                        pin_memory=device.type == "cuda", persistent_workers=cfg["data"]["num_workers"] > 0)
    batches = infinite(loader, sampler)
    val = val_subset(cfg) if main_proc else None
    eval_model = lambda: ema.model if ema else model  # noqa: E731  (EMA weights for validation)

    def hours():
        """Total training time over all sessions, in hours."""
        return round(hours_before + (time.time() - start) / 3600, 3)

    def save(tag="last"):
        state = training_state(step, model, ema, opt, sched, scaler, best, cfg, hours())
        save_checkpoint(out / "checkpoints" / f"{tag}.pt", state)

    last_save = last_val = time.time()
    last_val_step = step   # validation counts steps from here (no back-to-back validations)
    running, n_running, step_seconds = 0.0, 0, 0.0
    limit = tc["time_limit_hours"] * 3600
    val_seconds = tc.get("val_minutes", float("inf")) * 60
    model.train()
    while step < tc["max_steps"]:
        t0 = time.time()
        loss = train_step(net, model, ema, opt, sched, scaler, next(batches), device, use_fp16, tc["grad_clip"])
        step_seconds += time.time() - t0
        step += 1
        running, n_running = running + loss, n_running + 1

        if step % tc["log_every"] == 0:          # step-based: all processes take part
            mean_loss = dist.mean(running / n_running)
            samples = n_running * tc["batch_size"] * dist.world_size
            if main_proc:
                append_csv(out / "train_log.csv", {
                    "step": step, "loss": round(mean_loss, 6), "lr": f"{sched.get_last_lr()[0]:.3e}",
                    "samples_per_s": round(samples / step_seconds, 1), "gpus": dist.world_size,
                    "hours": hours()})
            log(f"step {step}: loss {mean_loss:.4f} ({samples / step_seconds:.1f} samples/s)")
            running, n_running, step_seconds = 0.0, 0, 0.0

        # Clock-based decisions: made by the main process, shared with all.
        now = time.time()
        do_val, do_save, do_stop = dist.broadcast_flags([
            step - last_val_step >= tc["val_every"] or step == tc["max_steps"] or now - last_val >= val_seconds,
            now - last_save >= tc["checkpoint_minutes"] * 60,
            now - start >= limit])
        if do_val:
            last_val, last_val_step = time.time(), step
            if main_proc:
                t_val = time.time()
                scores, preds = validate(eval_model(), val, device, steps=tc.get("val_ddim_steps"))
                append_csv(out / "val_log.csv", {"step": step, **scores, "hours": hours(),
                                                 "val_seconds": round(time.time() - t_val, 1)})
                save_grid(val, preds, out / "samples" / f"step_{step:07d}.png", tc["sample_grid_cases"])
                log(f"step {step}: val {scores}")
                if scores["tmIoU"] > best:
                    best = scores["tmIoU"]
                    torch.save({"step": step, "model": eval_model().state_dict(), "config": cfg,
                                "val": scores}, out / "checkpoints" / "best.pt")
            dist.barrier()                       # the other processes wait here
        if do_save:
            if main_proc:
                save()
            last_save = time.time()
        if do_stop:
            log(f"Time limit of {tc['time_limit_hours']} h reached at step {step}: saving and stopping.")
            break

    if main_proc:
        save()
    log(f"Stopped at step {step}; checkpoint saved to {last}")
    dist.barrier()
    dist.close()


if __name__ == "__main__":
    main()
