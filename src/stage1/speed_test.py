"""Measure training and validation speed on this machine's GPU.

For each model (U-Net baseline, diffusion) at full size:
  1. find the largest batch size from {32, 64, 128} that fits in GPU memory
     (3 real training steps each, fp16 + EMA as in training);
  2. time `--steps` training steps at that batch size with the real data
     pipeline (DataLoader + random corruptions);
  3. time the data pipeline alone (can it keep up with the GPU?);
  4. time one validation pass over the fixed val subset (diffusion: 50 DDIM steps).

    python -m src.stage1.speed_test --out /kaggle/working/speed_results.json
    python -m src.stage1.speed_test --smoke          # tiny models on CPU, to test the script

Needs data/processed/train.npz and the val benchmark (data/benchmarks/).
"""

import argparse
import gc
import json
import time

import torch
from torch.utils.data import DataLoader

from src.stage1.data.dataset import CorruptionDataset
from src.stage1.models import build_model
from src.stage1.models.backbone import count_parameters
from src.stage1.train import EMA, LossModule, infinite, load_settings, lr_factor, train_step, val_subset, validate
from src.utils import load_config, project_path

CONFIGS = {"unet": "configs/stage1/train_unet.yaml", "diffusion": "configs/stage1/train_diffusion.yaml"}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def make_trainer(cfg, device):
    """Loss module + model + EMA + optimiser + schedule + scaler, as in train.py (single GPU)."""
    tc = cfg["train"]
    model = build_model(cfg["model"]).to(device).train()
    ema = EMA(model, tc["ema_decay"]) if cfg["model"]["kind"] == "diffusion" else None
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, tc["warmup_steps"], tc["max_steps"]))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    return LossModule(model), model, ema, opt, sched, scaler


def free(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def try_batch_size(cfg, dataset, bs, device):
    """3 training steps at batch size bs. Returns peak GPU memory (GB) or None if it doesn't fit."""
    free(device)
    try:
        trainer = make_trainer(cfg, device)
        batch = next(iter(DataLoader(dataset, batch_size=bs, shuffle=True)))
        for _ in range(3):
            train_step(*trainer, batch, device, device.type == "cuda", cfg["train"]["grad_clip"])
        sync(device)
        peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
        return round(peak, 2)
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        trainer = batch = None  # noqa: F841  (release memory)
        free(device)


def measure(kind, args, device, dataset):
    cfg = load_settings(project_path(CONFIGS[kind]), args.smoke)
    result = {"model": kind}
    sizes = [2, 4] if args.smoke else [32, 64, 128]

    # 1. Largest batch size that fits.
    fits = {}
    for bs in sizes:
        peak = try_batch_size(cfg, dataset, bs, device)
        fits[bs] = peak
        print(f"[{kind}] batch {bs}: {'fits, peak %.2f GB' % peak if peak is not None else 'OUT OF MEMORY'}")
        if peak is None:
            break
    ok = [bs for bs, peak in fits.items() if peak is not None]
    if not ok:
        result["error"] = "no batch size fits"
        return result
    bs = max(ok)
    result.update(batch_size=bs, peak_memory_gb_per_batch_size={str(k): v for k, v in fits.items()})

    # 2. Timed training steps with the real data pipeline.
    free(device)
    trainer = make_trainer(cfg, device)
    result["parameters_M"] = round(count_parameters(trainer[1]) / 1e6, 2)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True,
                        num_workers=cfg["data"]["num_workers"], pin_memory=device.type == "cuda",
                        persistent_workers=cfg["data"]["num_workers"] > 0)
    batches = infinite(loader)
    for _ in range(10):                      # warm-up (workers start, CUDA kernels compile)
        train_step(*trainer, next(batches), device, device.type == "cuda", cfg["train"]["grad_clip"])
    sync(device)
    t = time.time()
    for _ in range(args.steps):
        train_step(*trainer, next(batches), device, device.type == "cuda", cfg["train"]["grad_clip"])
    sync(device)
    dt = time.time() - t
    result.update(steps_per_second=round(args.steps / dt, 3), samples_per_second=round(args.steps * bs / dt, 1),
                  seconds_per_step=round(dt / args.steps, 4))

    # 3. Data pipeline alone.
    t = time.time()
    n = 30
    for _ in range(n):
        next(batches)
    result["data_only_samples_per_second"] = round(n * bs / (time.time() - t), 1)

    # 4. One validation pass (EMA weights for diffusion, as in training).
    model, ema = trainer[1], trainer[2]
    val = val_subset(cfg)
    sync(device)
    t = time.time()
    validate(ema.model if ema else model, val, device, steps=cfg["train"].get("val_ddim_steps"))
    sync(device)
    result.update(val_cases=len(val["cond"]), val_seconds=round(time.time() - t, 1),
                  ddim_steps=cfg["model"].get("ddim_steps"))
    print(f"[{kind}] {json.dumps(result)}")
    trainer = loader = batches = None  # noqa: F841
    free(device)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--smoke", action="store_true", help="tiny models (CPU test of this script)")
    parser.add_argument("--out", default=None, help="write results JSON here")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = {"device": str(device), "torch": torch.__version__}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        info.update(gpu=torch.cuda.get_device_name(0), gpu_count=torch.cuda.device_count(),
                    gpu_memory_gb=round(props.total_memory / 1e9, 1))
    print(json.dumps(info))
    dataset = CorruptionDataset(project_path("data/processed/train.npz"), load_config("stage1/corruption"))
    results = {"environment": info, "models": [measure(k, args, device, dataset) for k in CONFIGS]}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
