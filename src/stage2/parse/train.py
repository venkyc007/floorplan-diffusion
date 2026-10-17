"""Train the Stage 2 parser (drawing -> 16-class map): U-Net with an ImageNet ResNet-34 encoder.

    python -m src.stage2.parse.train --config configs/stage2/parser_v1.yaml \
        --data <dir with train/ and val/ parser folders> --out <run dir> [--freq stats.npz] [--smoke]
    (2 GPUs: torchrun --standalone --nproc_per_node=2 -m src.stage2.parse.train ...)

  * AdamW: decoder/head lr 3e-4, encoder 1e-4; 500 warm-up steps, then cosine decay over
    the session's TIME budget (the run is stopped by time_limit_hours, not by a step count);
  * mixed precision (AMP) on GPU, gradient clipping 1.0, DDP over all GPUs (src/distributed.py);
  * last.pt every 30 min and at the end; automatic resume from <out>/checkpoints/last.pt;
  * every ~20 min: validation on the FULL val drawings (whole image, padded to a multiple
    of 32) -> val_log.csv; best.pt by mIoU; prints the CubiCasa fallback trigger.
"""

import argparse
import csv
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

from src.distributed import Distributed
from src.stage2.data.classes16 import C16
from src.stage2.parse.dataset import ParserTrainSet, list_plans, pad_to_multiple, read_pair
from src.stage2.parse.losses import ParserLoss, class_weights
from src.stage2.parse.metrics import ParserMetrics, fallback_check

GREY_MEAN, GREY_STD = 0.449, 0.226      # ImageNet mean/std averaged over RGB (the encoder's 1-channel conv sums RGB)


class Parser(torch.nn.Module):
    """smp U-Net with input normalisation built in: takes greyscale in [0, 1], returns 16 logits."""

    def __init__(self, mcfg, encoder_weights="config"):
        super().__init__()
        import segmentation_models_pytorch as smp
        weights = mcfg["encoder_weights"] if encoder_weights == "config" else encoder_weights
        self.net = getattr(smp, mcfg["arch"])(encoder_name=mcfg["encoder"], encoder_weights=weights,
                                              in_channels=mcfg["in_channels"], classes=mcfg["classes"])

    def forward(self, x):
        return self.net((x - GREY_MEAN) / GREY_STD)


def load_cfg(path, smoke):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if smoke:
        s = cfg["smoke"]
        cfg["data"]["crop"] = s["crop"]
        cfg["train"].update(batch_size=s["batch_size"], workers=s["workers"])
        cfg["model"]["encoder_weights"] = s["encoder_weights"]
    return cfg


def class_freq(path):
    """Train-split parser pixel frequencies from the data kernel's report/stats.npz (uniform if missing)."""
    if path and Path(path).exists():
        st = np.load(path)
        pix = st["pix_parser"][st["split"] == "train"].sum(0)
        return pix / pix.sum()
    return np.full(C16.n, 1.0 / C16.n)


def lr_factor(step, progress, warmup):
    """Linear warm-up over `warmup` steps, then cosine over the time budget (progress 0..1)."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def append_csv(path, row):
    new = not Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


@torch.no_grad()
def validate(model, folder, cfg, device, limit=None, amp=True):
    """Whole-drawing validation -> metrics dict."""
    model.eval()
    vcfg = cfg["val"]
    m = ParserMetrics(vcfg["component_classes"], vcfg["component_iou"], cfg["data"]["ignore_index"])
    plans = list_plans(folder)[:limit] if limit else list_plans(folder)
    for plan in plans:
        img, lbl = read_pair(folder, plan)
        pi, _ = pad_to_multiple(img, lbl, vcfg["pad_multiple"], cfg["data"]["pad_image"], cfg["data"]["ignore_index"])
        x = torch.from_numpy(pi)[None, None].float().to(device) / 255
        pred = predict_safe(model, x, amp and device.type == "cuda")[: img.shape[0], : img.shape[1]]
        m.update(pred, lbl)
    model.train()
    res = m.result()
    res["oom_tiled"] = OOM_COUNT[0]
    return res


OOM_COUNT = [0]     # drawings that needed the tiled fallback (logged in val_log.csv)


def predict_safe(model, x, amp, tile=1024, overlap=64):
    """Whole-drawing prediction (1, 1, H, W) -> (H, W) labels. OOM safeguard: if the whole
    drawing does not fit in GPU memory, predict it in overlapping tiles instead."""
    try:
        with torch.autocast(device_type=x.device.type, enabled=amp):
            return model(x).argmax(1)[0].cpu().numpy()
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        OOM_COUNT[0] += 1
    H, W = x.shape[-2:]
    out = torch.zeros(H, W, dtype=torch.long)
    step = tile - 2 * overlap
    for y in range(0, H, step):
        for x0 in range(0, W, step):
            ya, xa = max(0, min(y - overlap, H - tile)), max(0, min(x0 - overlap, W - tile))
            t = x[..., ya:ya + tile, xa:xa + tile]
            with torch.autocast(device_type=x.device.type, enabled=amp):
                p = model(t).argmax(1)[0].cpu()
            yb, xb = min(y + step, H), min(x0 + step, W)
            out[y:yb, x0:xb] = p[y - ya:yb - ya, x0 - xa:xb - xa]
    return out.numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="folder holding train/ and val/ (extracted parser zips)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--freq", default=None, help="data kernel report/stats.npz (class frequencies)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    D = Distributed()
    cfg = load_cfg(args.config, args.smoke)
    tc = cfg["train"]
    out = Path(args.out)
    ckpt_dir = out / "checkpoints"
    if D.is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    torch.manual_seed(tc["seed"] + D.rank)
    device = D.device
    amp = tc["amp"] and device.type == "cuda"

    model = Parser(cfg["model"]).to(device)
    enc = list(model.net.encoder.parameters())
    enc_ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc, "lr": tc["lr_encoder"]}, {"params": rest, "lr": tc["lr"]}],
                            weight_decay=tc["weight_decay"])
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    crit = ParserLoss(class_weights(class_freq(args.freq), cfg["loss"]["ce_weight_cap"]),
                      cfg["loss"]["dice"], cfg["data"]["ignore_index"]).to(device)

    step, best, hours_done = 0, -1.0, 0.0
    last = ckpt_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, best, hours_done = ck["step"], ck["best"], ck["hours"]
        torch.manual_seed(tc["seed"] + D.rank + step)
        if D.is_main:
            print(f"resumed from step {step} ({hours_done:.2f} h trained, best mIoU {best:.4f})", flush=True)
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[D.local_rank] if device.type == "cuda" else None) \
        if D.enabled else model

    ds = ParserTrainSet(Path(args.data) / "train", cfg)
    sampler = torch.utils.data.RandomSampler(ds, replacement=True, num_samples=10**8,
                                             generator=torch.Generator().manual_seed(tc["seed"] + D.rank + step))
    loader = torch.utils.data.DataLoader(ds, batch_size=tc["batch_size"], sampler=sampler, num_workers=tc["workers"],
                                         pin_memory=device.type == "cuda", drop_last=True,
                                         persistent_workers=tc["workers"] > 0)
    budget = tc["time_limit_hours"]
    session_start = time.time()
    last_ckpt = last_val = time.time()
    smoke = cfg["smoke"] if args.smoke else None

    def hours():
        return hours_done + (time.time() - session_start) / 3600

    def save(path, tag):
        state = {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                 "step": step, "best": best, "hours": hours(), "config": cfg, "tag": tag}
        tmp = Path(str(path) + ".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)

    if D.is_main:
        print(f"parser: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M params, {len(ds.plans)} train plans, "
              f"{D.world_size} process(es), batch {tc['batch_size']}/process", flush=True)
    t_log, n_log, losses = time.time(), 0, []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # time-based cosine: progress = trained hours / session budget (broadcast so all ranks agree)
        prog = torch.tensor([hours() / budget], device=device)
        if D.enabled:
            dist.broadcast(prog, src=0)
        f = lr_factor(step, float(prog), tc["warmup_steps"])
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = net(x)
        loss, parts = crit(logits, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        scaler.step(opt)
        scaler.update()
        step += 1
        losses.append(float(loss))
        n_log += x.shape[0]

        if step % (1 if smoke else tc["log_every"]) == 0:
            lm = D.mean(np.mean(losses))
            if D.is_main:
                sps = n_log * D.world_size / (time.time() - t_log)
                print(f"step {step}: loss {lm:.4f} (ce {parts['ce']:.3f}, dice {parts['dice']:.3f}) "
                      f"lr x{f:.3f} {sps:.1f} samples/s {hours():.2f} h", flush=True)
                append_csv(out / "train_log.csv", {"step": step, "hours": round(hours(), 4), "loss": round(lm, 5),
                                                   "ce": round(parts["ce"], 5), "dice": round(parts["dice"], 5),
                                                   "lr_factor": round(f, 4), "samples_per_s": round(sps, 2),
                                                   "gpus": D.world_size})
            t_log, n_log, losses = time.time(), 0, []

        now = time.time()
        stop = (time.time() - t_start) / 3600 >= budget - 0.08 or hours() >= budget
        if smoke:
            do_val, do_ckpt, stop = step % smoke["val_steps"] == 0, step % smoke["checkpoint_steps"] == 0, \
                step >= smoke["max_steps"]
        else:
            do_val = now - last_val >= tc["val_minutes"] * 60
            do_ckpt = now - last_ckpt >= tc["checkpoint_minutes"] * 60
        do_val, do_ckpt, stop = D.broadcast_flags([do_val, do_ckpt, stop])
        if do_val or stop:
            if D.is_main:
                t0 = time.time()
                res = validate(model, Path(args.data) / "val", cfg, device,
                               limit=smoke["val_plans"] if smoke else None, amp=amp)
                ok = fallback_check(res, cfg["val"]["fallback"])
                print(f"VAL step {step}: mIoU {res['mIoU']:.4f} | wall IoU {res['iou_wall']:.3f} | F1 door "
                      f"{res['F1_door']:.3f} window {res['F1_window']:.3f} toilet {res['F1_toilet']:.3f} sink "
                      f"{res['F1_sink']:.3f} bath {res['F1_bath_shower']:.3f} appliance {res['F1_appliance']:.3f} | "
                      f"fallback trigger: {'NOT needed' if ok else 'TRIGGERED'} ({time.time() - t0:.0f} s)", flush=True)
                append_csv(out / "val_log.csv", {"step": step, "hours": round(hours(), 4),
                                                 **{k: round(v, 5) for k, v in res.items()}, "fallback_ok": ok})
                if res["mIoU"] > best:
                    best = res["mIoU"]
                    save(ckpt_dir / "best.pt", "best")
            last_val = time.time()
            D.barrier()
        if do_ckpt or stop:
            if D.is_main:
                save(last, "last")
            last_ckpt = time.time()
            D.barrier()
        if stop:
            break
    if D.is_main:
        print(f"stopped at step {step}, {hours():.2f} h; best val mIoU {best:.4f}", flush=True)
    D.close()


if __name__ == "__main__":
    main()
