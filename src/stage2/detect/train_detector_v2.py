"""Detector v2: the missing-component detector trained on what it really sees (user, 2026-10-19).

    python -m src.stage2.detect.train_detector_v2 --config configs/stage2/detector_v2.yaml --data <completion zips dir> \
        --parsed parsed_train.zip --val-parsed parsed_val.zip --init <detector v1 best.pt> --out <run dir> [--smoke]

Same model, inputs (one-hot plan + violation maps + R1/R3 maps), outputs (MISSING types) and loss as detector v1
(src/stage2/detect/train_detector.py); warm-started from detector v1. Training windows (128 x 128):
  70%  parser v2B maps of train drawings with image-level omissions (make_parsed_set.py; target = the omission's
       type on its pixels; intact versions give all-zero targets). Half of the windows of an omitted version are
       centred on the omission, the rest on a random building pixel.
  30%  the old label-level simulation (ground-truth maps + omissions.simulate), for variety.
Validation: 512 fixed windows of the parsed TUNING drawings (same recipe), DetMetrics as v1; best.pt by mean F1.
CE class weights from 300 sampled training windows of the mix. AdamW, warm-up then cosine over the time budget.
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
import yaml

from src.stage2.complete.data16 import load_plans
from src.stage2.data.classes16 import C16
from src.stage2.detect.omissions import simulate
from src.stage2.detect.train_detector import (N_OUT, DetMetrics, append_csv, build_model, features, random_window,
                                              rule_maps)
from src.stage2.parse.losses import ParserLoss, class_weights

BG = C16.index["background"]


def load_parsed(path):
    """[(map png bytes, target png bytes, omitted?)] from a make_parsed_set zip."""
    z = zipfile.ZipFile(path)
    names = sorted(n[:-8] for n in z.namelist() if n.endswith("_map.png"))
    return [(z.read(f"{n}_map.png"), z.read(f"{n}_tgt.png"), not n.endswith("_v0")) for n in names]


def dec(b):
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)


def parsed_window(item, rng, win):
    lab, tgt = dec(item[0]), dec(item[1])
    pts = np.argwhere(tgt > 0) if item[2] and rng.random() < 0.5 else np.argwhere(lab != BG)
    if not len(pts):
        pts = np.argwhere(lab != BG)
    cy, cx = pts[rng.integers(len(pts))] if len(pts) else (lab.shape[0] // 2, lab.shape[1] // 2)
    out_l = np.full((win, win), BG, np.uint8)
    out_t = np.zeros((win, win), np.uint8)
    y0, x0 = cy - win // 2, cx - win // 2
    h, w = lab.shape
    sy0, sx0, sy1, sx1 = max(0, y0), max(0, x0), min(h, y0 + win), min(w, x0 + win)
    out_l[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = lab[sy0:sy1, sx0:sx1]
    out_t[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = tgt[sy0:sy1, sx0:sx1]
    if rng.random() < 0.5:
        out_l, out_t = out_l[:, ::-1], out_t[:, ::-1]
    k = int(rng.integers(4))
    return np.ascontiguousarray(np.rot90(out_l, k)), np.ascontiguousarray(np.rot90(out_t, k))


class MixSet(torch.utils.data.Dataset):
    def __init__(self, parsed, plans, cfg, share, length=10 ** 7, fixed_seed=None):
        self.parsed, self.plans, self.cfg, self.share = parsed, plans, cfg, share
        self.length, self.fixed, self.rng = length, fixed_seed, None

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.fixed is not None:
            rng = np.random.default_rng(self.fixed + i)
        else:
            if self.rng is None:
                self.rng = np.random.default_rng(torch.initial_seed() % 2 ** 32)
            rng = self.rng
        d = self.cfg["data"]
        if not self.plans or rng.random() < self.share:
            plan, target = parsed_window(self.parsed[rng.integers(len(self.parsed))], rng, d["window"])
        else:
            lab = random_window(self.plans[rng.integers(len(self.plans))][1], rng, d["window"])
            plan, target, _ = simulate(lab, rng, d["intact_p"], tuple(d["n_omit"]))
        return torch.from_numpy(plan.astype(np.int64)), torch.from_numpy(rule_maps(plan)), \
            torch.from_numpy(target.astype(np.int64))


@torch.no_grad()
def validate(model, loader, device, K):
    model.eval()
    m = DetMetrics()
    for plan, rules, target in loader:
        pred = model(features(plan.to(device), rules, K)).argmax(1).cpu().numpy()
        for p, t in zip(pred, target.numpy()):
            m.update(p, t)
    model.train()
    return m.result()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--parsed", required=True)
    ap.add_argument("--val-parsed", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
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
    parsed, vparsed = load_parsed(args.parsed), load_parsed(args.val_parsed)
    plans = load_plans(Path(args.data) / "completion_train.zip", limit=20 if args.smoke else None, seed=0)
    share = cfg["data"]["parsed_share"]
    ds = MixSet(parsed, plans, cfg, share)
    vds = MixSet(vparsed, [], cfg, 1.0, length=8 if args.smoke else cfg["val"]["windows"], fixed_seed=12345)
    loader = torch.utils.data.DataLoader(ds, batch_size=tc["batch_size"], num_workers=tc["workers"],
                                         persistent_workers=tc["workers"] > 0, drop_last=True)
    vloader = torch.utils.data.DataLoader(vds, batch_size=tc["batch_size"], num_workers=tc["workers"])
    cds = MixSet(parsed, plans, cfg, share, length=40 if args.smoke else 300, fixed_seed=777)
    counts = np.zeros(N_OUT)
    for i in range(len(cds)):
        counts += np.bincount(cds[i][2].numpy().ravel(), minlength=N_OUT)
    crit = ParserLoss(class_weights(counts / counts.sum(), cfg["loss"]["ce_weight_cap"]), cfg["loss"]["dice"]).to(device)

    ck = torch.load(args.init, map_location=device, weights_only=False)
    cfg["model"] = ck["config"]["model"]                     # same architecture as v1
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ck["model"])
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
    K = cfg["model"]["K"]
    budget, session = tc["time_limit_hours"], time.time()

    def hours():
        return hours_done + (time.time() - session) / 3600

    def save(path, tag):
        st = {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step,
              "best": best, "hours": hours(), "config": cfg, "tag": tag, "class_weights": crit.weights.cpu().tolist()}
        torch.save(st, str(path) + ".tmp")
        os.replace(str(path) + ".tmp", path)

    if step == 0:                                            # detector v1 on the same parsed val windows
        res = validate(model, vloader, device, K)
        append_csv(out / "val_log.csv", {"step": 0, "hours": 0.0, **{k: round(float(v), 5) for k, v in res.items()}})
        print(f"VAL v1: mean F1 {res['mean_F1']:.3f} false alarms {res['false_alarm_rate']:.3f}", flush=True)
    print(f"detector v2: {len(parsed)} parsed versions, {len(plans)} label plans, share {share}, "
          f"class weights {[round(float(w), 2) for w in crit.weights]}", flush=True)
    last_ckpt = last_val = time.time()
    t_log, n_log, losses = time.time(), 0, []
    for plan, rules, target in loader:
        x = features(plan.to(device), rules, K)
        y = target.to(device)
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
            res = validate(model, vloader, device, K)
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


if __name__ == "__main__":
    main()
