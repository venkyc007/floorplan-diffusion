"""Train the missing-component detector (spec section 6, "both" = rules + this learned part).

    python -m src.stage2.detect.train_detector --config configs/stage2/detector.yaml \
        --data <folder with completion_train.zip, completion_val.zip> --out <run dir> [--smoke]

Input (33 channels, 128 x 128 windows of completion-scale plans):
  16  one-hot plan (after simulated omissions + parser-like noise, omissions.py)
  15  absolute-mode violation maps of every constraint that needs no hole
      (C1 C3 C4 C6 C7 C8 C9 C10 C11 C12 C13 C14 C15 C16 C18; computed on the GPU per batch)
   2  pixel maps of the check-only rules R1 (room area out of range, rooms not cut by the window
      border) and R3 (door joining a forbidden pair)
Output: per-pixel missing type (10: none, wall, door, window, railing, room, 4 fixtures).
Loss: class-weighted CE (sqrt(f_max / f_c), cap 10, frequencies from 300 sampled windows) + Dice.
Validation (fixed simulated val windows): per-type pixel P/R/F1, share of omitted components
detected (>= 30% of their pixels predicted as that type), false alarms = intact windows with
>= 20 predicted missing pixels. best.pt by mean pixel F1 over the 9 missing types.
Training: AdamW, warm-up then cosine over the time budget, AMP, grad clip 1.0, last.pt every
30 min + auto-resume, stops at time_limit_hours.
"""

import argparse
import csv
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from src.stage2.complete.data16 import load_plans
from src.stage2.constraints.constraints import CONST, Ctx, one_hot, violation_map
from src.stage2.constraints.rules import components, r3_adjacency, room_instances
from src.stage2.data.classes16 import C16
from src.stage2.detect.omissions import MISSING, simulate
from src.stage2.parse.losses import ParserLoss, class_weights

FEATURE_CONSTRAINTS = ["C1", "C3", "C4", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C18"]
N_IN = C16.n + len(FEATURE_CONSTRAINTS) + 2
N_OUT = len(MISSING)


def random_window(label, rng, win):
    """A win x win crop centred on a random non-background pixel (background-padded)."""
    inside = np.argwhere(label != C16.index["background"])
    cy, cx = inside[rng.integers(len(inside))] if len(inside) else (label.shape[0] // 2, label.shape[1] // 2)
    out = np.full((win, win), C16.index["background"], np.uint8)
    y0, x0 = cy - win // 2, cx - win // 2
    h, w = label.shape
    sy0, sx0, sy1, sx1 = max(0, y0), max(0, x0), min(h, y0 + win), min(w, x0 + win)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = label[sy0:sy1, sx0:sx1]
    if rng.random() < 0.5:
        out = out[:, ::-1]
    return np.ascontiguousarray(np.rot90(out, int(rng.integers(4))))


def rule_maps(label, const=CONST):
    """(2, H, W) float: R1 (out-of-range room instances not touching the border), R3 (forbidden doors)."""
    r1 = np.zeros(label.shape, np.float32)
    ranges = const.get("room_area", {})
    for c, m in room_instances(label):
        if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
            continue
        lo_hi = ranges.get(C16.names[c])
        if lo_hi and not (lo_hi[0] <= m.sum() <= lo_hi[1]):
            r1[m] = 1
    r3 = r3_adjacency(label, const)[1].astype(np.float32)
    return np.stack([r1, r3])


class DetectorSet(torch.utils.data.Dataset):
    def __init__(self, plans, cfg, length=10**7, fixed_seed=None):
        self.plans, self.cfg, self.length, self.fixed = plans, cfg, length, fixed_seed
        self.rng = None

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.fixed is not None:
            rng = np.random.default_rng(self.fixed + i)
        else:
            if self.rng is None:
                self.rng = np.random.default_rng(torch.initial_seed() % 2**32)
            rng = self.rng
        d = self.cfg["data"]
        lab = random_window(self.plans[rng.integers(len(self.plans))][1], rng, d["window"])
        plan, target, _ = simulate(lab, rng, d["intact_p"], tuple(d["n_omit"]))
        return torch.from_numpy(plan.astype(np.int64)), torch.from_numpy(rule_maps(plan)), \
            torch.from_numpy(target.astype(np.int64))


@torch.no_grad()
def features(plan, rules, K=96):
    """(B, H, W) labels + (B, 2, H, W) rule maps -> (B, 33, H, W) network input (on plan's device)."""
    P = one_hot(plan)
    ctx = Ctx(const=CONST, K=K)
    maps = [violation_map(n, P, ctx).clamp(0, 1) for n in FEATURE_CONSTRAINTS]
    return torch.cat([P] + maps + [rules.to(P)], 1)


def build_model(mcfg):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=mcfg["encoder"], encoder_weights=None, in_channels=N_IN, classes=N_OUT)


class DetMetrics:
    def __init__(self):
        self.tp, self.fp, self.fn = (np.zeros(N_OUT) for _ in range(3))
        self.comp_hit, self.comp_all = np.zeros(N_OUT), np.zeros(N_OUT)
        self.intact, self.false_alarm = 0, 0

    def update(self, pred, target):
        for t in range(1, N_OUT):
            p, g = pred == t, target == t
            self.tp[t] += (p & g).sum()
            self.fp[t] += (p & ~g).sum()
            self.fn[t] += (~p & g).sum()
            for m in components(g):
                self.comp_all[t] += 1
                self.comp_hit[t] += (p[m].mean() >= 0.3)
        if not target.any():
            self.intact += 1
            self.false_alarm += int((pred > 0).sum() >= 20)

    def result(self):
        out, f1s = {}, []
        for t in range(1, N_OUT):
            prec = self.tp[t] / max(self.tp[t] + self.fp[t], 1)
            rec = self.tp[t] / max(self.tp[t] + self.fn[t], 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            name = MISSING[t]
            out.update({f"P_{name}": prec, f"R_{name}": rec, f"F1_{name}": f1,
                        f"detected_{name}": self.comp_hit[t] / max(self.comp_all[t], 1)})
            f1s.append(f1)
        out["mean_F1"] = float(np.mean(f1s))
        out["false_alarm_rate"] = self.false_alarm / max(self.intact, 1)
        return out


@torch.no_grad()
def validate(model, loader, device, K):
    model.eval()
    m = DetMetrics()
    for plan, rules, target in loader:
        x = features(plan.to(device), rules, K)
        pred = model(x).argmax(1).cpu().numpy()
        for p, t in zip(pred, target.numpy()):
            m.update(p, t)
    model.train()
    return m.result()


def append_csv(path, row):
    new = not Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tc = cfg["train"]
    if args.smoke:
        tc = tc | {"batch_size": 2, "workers": 0, "warmup_steps": 2, "log_every": 1}
        cfg["model"]["encoder"] = "resnet18"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    torch.manual_seed(tc["seed"])
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    data = Path(args.data)
    plans = load_plans(data / "completion_train.zip", limit=20 if args.smoke else None, seed=0)
    vplans = load_plans(data / "completion_val.zip", limit=20 if args.smoke else None, seed=0)
    ds = DetectorSet(plans, cfg)
    n_val = 8 if args.smoke else cfg["val"]["windows"]
    vds = DetectorSet(vplans, cfg, length=n_val, fixed_seed=12345)
    loader = torch.utils.data.DataLoader(ds, batch_size=tc["batch_size"], num_workers=tc["workers"],
                                         persistent_workers=tc["workers"] > 0, drop_last=True)
    vloader = torch.utils.data.DataLoader(vds, batch_size=tc["batch_size"], num_workers=tc["workers"])

    # class weights from sampled windows
    rng_ds = DetectorSet(plans, cfg, length=40 if args.smoke else 300, fixed_seed=777)
    counts = np.zeros(N_OUT)
    for i in range(len(rng_ds)):
        counts += np.bincount(rng_ds[i][2].numpy().ravel(), minlength=N_OUT)
    crit = ParserLoss(class_weights(counts / counts.sum(), cfg["loss"]["ce_weight_cap"]), cfg["loss"]["dice"]).to(device)

    model = build_model(cfg["model"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step, best, hours_done = 0, -1.0, 0.0
    last = out / "checkpoints" / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, best, hours_done = ck["step"], ck["best"], ck["hours"]
        print(f"resumed at step {step} ({hours_done:.2f} h, best mean F1 {best:.4f})", flush=True)
    budget, session = tc["time_limit_hours"], time.time()
    last_ckpt = last_val = time.time()
    K = cfg["model"]["K"]

    def hours():
        return hours_done + (time.time() - session) / 3600

    def save(path, tag):
        st = {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step,
              "best": best, "hours": hours(), "config": cfg, "tag": tag,
              "class_weights": crit.weights.cpu().tolist()}
        torch.save(st, str(path) + ".tmp")
        os.replace(str(path) + ".tmp", path)

    print(f"detector: in {N_IN} ch, out {N_OUT}, {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M params, "
          f"{len(plans)} plans, class weights {[round(float(w), 2) for w in crit.weights]}", flush=True)
    t_log, n_log, losses = time.time(), 0, []
    for plan, rules, target in loader:
        x = features(plan.to(device), rules, K)
        y = target.to(device)
        prog = hours() / budget
        f = (step + 1) / tc["warmup_steps"] if step < tc["warmup_steps"] else 0.5 * (1 + math.cos(math.pi * min(prog, 1)))
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
            print(f"step {step}: loss {np.mean(losses):.4f} (ce {parts['ce']:.3f} dice {parts['dice']:.3f}) "
                  f"lr x{f:.3f} {sps:.1f} samples/s {hours():.2f} h", flush=True)
            append_csv(out / "train_log.csv", {"step": step, "hours": round(hours(), 4), "loss": round(float(np.mean(losses)), 5),
                                               "samples_per_s": round(sps, 2)})
            t_log, n_log, losses = time.time(), 0, []
        now = time.time()
        stop = hours() >= budget or (now - t_start) / 3600 >= budget - 0.05 or (args.smoke and step >= 4)
        if now - last_val >= tc["val_minutes"] * 60 or stop or (args.smoke and step % 2 == 0):
            res = validate(model, vloader, device, K)
            print(f"VAL step {step}: mean F1 {res['mean_F1']:.3f} false alarms {res['false_alarm_rate']:.3f} | " +
                  " ".join(f"{n} F1 {res['F1_' + n]:.2f}/det {res['detected_' + n]:.2f}" for n in MISSING[1:]), flush=True)
            append_csv(out / "val_log.csv", {"step": step, "hours": round(hours(), 4), **{k: round(float(v), 5) for k, v in res.items()}})
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


if __name__ == "__main__":
    main()
