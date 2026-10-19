"""Parser v2: fine-tune parser v1 with text erasure + the local constraint losses (one GPU per variant).

    python -m src.stage2.parse.train_v2 --config configs/stage2/parser_v2.yaml --variant A \
        --data <dir with train/ val/> --init <parser v1 best.pt> --out <run dir> --freq stats.npz \
        --tuning tuning_ids.txt [--smoke]

Variants (user, 2026-10-19):
  A "no text"  text erased on 100% of training crops; at inference text is ALWAYS erased first (parse_v2).
  B "mixed"    text erased on 50% of training crops; normal inference.
Text erasure: src/stage2/parse/text_erase.py (whole words, ground-truth labels), applied to the crop at the
drawing's own scale, before resizing and the photometric augmentations.

Loss = CE + Dice (as v1) + sum_i w_i * L_i over C3, C6, C9, C10, C11, C12, C15, C16. The constraints are
defined at completion scale (walls 3 px), so each sample's softmax is shrunk (area interpolation) by
3 / (8 * crop scale) before they are computed; the labels are shrunk (nearest) for the GT-hinged ones.
Samples with ignore pixels (untyped plans) get no constraint loss. Modes as in the registry.
Weights by GRADIENT SIZE (spec section 3): over the first `calib_batches` batches, G_0 = ||d(CE+Dice)/d logits||
and G_i = ||d L_i / d logits||; w_i = rho * median(G_0) / median(G_i), rho = min(0.05, 0.4 / n), capped at
10x the median weight. Logged to <out>/constraint_weights.yaml.
Checkpoint selection on TUNING val drawings only (the detector's tuning plans; the held-out plans stay
untouched): v2A by mIoU on text-erased tuning drawings, v2B by the mean of normal and text-erased mIoU.
Time-limited (cosine over the budget), last.pt every 30 min, automatic resume.
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

from src.stage2.constraints.constraints import Ctx, constraint_loss, one_hot
from src.stage2.data.classes16 import C16
from src.stage2.parse import augment
from src.stage2.parse.dataset import BACKGROUND, crop_around, list_plans, pad_to_multiple, read_pair
from src.stage2.parse.losses import ParserLoss, class_weights
from src.stage2.parse.metrics import ParserMetrics
from src.stage2.parse.text_erase import erase_text, text_mask
from src.stage2.parse.train import Parser, append_csv, class_freq, lr_factor, predict_safe
from src.stage2.data.erasure import erase

CONSTRAINTS = ["C3", "C6", "C9", "C10", "C11", "C12", "C15", "C16"]
ROOMS = C16.group("S")


# ---------------------------------------------------------------------------------------------- data

def one_crop(img, lbl, rng, cd, ca, crop, text_p):
    """As dataset._one_crop, plus text erasure at the drawing's scale. -> (image, labels, scale)."""
    scale = rng.uniform(*ca["scale"])
    src = max(8, int(round(crop / scale)))
    rare = [C16.index[c] for c in cd["rare_classes"]]
    centre = None
    if rng.random() < cd["rare_centre_prob"]:
        present = [c for c in rare if (lbl == c).any()]
        if present:
            ys, xs = np.nonzero(lbl == present[rng.integers(len(present))])
            i = rng.integers(len(ys))
            centre = (int(ys[i]), int(xs[i]))
    if centre is None:
        centre = (int(rng.integers(img.shape[0])), int(rng.integers(img.shape[1])))
    ci, cl = crop_around(img, lbl, *centre, src, cd["pad_image"])
    if rng.random() < text_p:
        ci = erase_text(ci, cl)
    ci = cv2_resize(ci, crop, src > crop)
    cl = cv2_resize(cl, crop, None)
    ci, cl = augment.rot_flip(ci, cl, rng, ca["rot90"], ca["flip"])
    return augment.photometric(ci, rng, ca), cl, scale


def cv2_resize(a, size, shrink):
    import cv2
    interp = cv2.INTER_NEAREST if shrink is None else (cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    return cv2.resize(a, (size, size), interpolation=interp)


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, folder, cfg, text_p, crop=None):
        self.folder, self.cfg, self.text_p = folder, cfg, text_p
        self.plans = list_plans(folder)
        self.crop = crop or cfg["data"]["crop"]
        self.rng = None

    def __len__(self):
        return 10 ** 7

    def __getitem__(self, i):
        if self.rng is None:
            self.rng = np.random.default_rng(torch.initial_seed() % 2 ** 32)
        img, lbl = read_pair(self.folder, self.plans[i % len(self.plans)])
        cd = self.cfg["data"]
        for _ in range(cd.get("max_tries", 10)):
            x, y, s = one_crop(img, lbl, self.rng, cd, self.cfg["augment"], self.crop, self.text_p)
            if (y != BACKGROUND).mean() >= cd.get("min_foreground", 0.0):
                break
        return torch.from_numpy(x)[None].float() / 255, torch.from_numpy(y.astype(np.int64)), torch.tensor(s)


# ---------------------------------------------------------------------------------------- constraints

def constraint_terms(logits, y, scales, ignore=255):
    """{name: mean loss over the usable samples} at completion scale (walls 3 px)."""
    P = logits.float().softmax(1)
    out = {n: [] for n in CONSTRAINTS}
    for b in range(len(P)):
        if (y[b] == ignore).any():
            continue
        f = 3.0 / (8.0 * float(scales[b]))
        size = max(32, int(round(P.shape[-1] * f)))
        Pb = F.interpolate(P[b:b + 1], size=(size, size), mode="area")
        yb = F.interpolate(y[b:b + 1, None].float(), size=(size, size), mode="nearest")[:, 0].long()
        ctx = Ctx(Y=one_hot(yb))
        for n in CONSTRAINTS:
            out[n].append(constraint_loss(n, Pb, ctx).mean())
    return {n: torch.stack(v).mean() for n, v in out.items() if v}


def calibrate(model, loader, crit, device, batches, amp):
    G = {"rec": []} | {n: [] for n in CONSTRAINTS}
    it = iter(loader)
    for _ in range(batches):
        x, y, s = next(it)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
        logits = logits.float().detach().requires_grad_(True)
        rec, _ = crit(logits, y)
        g, = torch.autograd.grad(rec, logits)
        G["rec"].append(float(g.norm()))
        for n, L in constraint_terms(logits, y, s).items():
            if not L.requires_grad:
                continue
            g, = torch.autograd.grad(L, logits, allow_unused=True, retain_graph=True)
            if g is not None and float(g.norm()) > 0:
                G[n].append(float(g.norm()))
    rho = min(0.05, 0.4 / len(CONSTRAINTS))
    g0 = float(np.median(G["rec"]))
    w = {n: rho * g0 / float(np.median(v)) for n, v in G.items() if n != "rec" and v}
    cap = 10 * float(np.median(list(w.values())))
    w = {n: min(v, cap) for n, v in w.items()}
    return w, {n: (float(np.median(v)) if v else 0.0, len(v)) for n, v in G.items()}


# ---------------------------------------------------------------------------------------- inference

def predict(model, img, device, amp):
    pi, _ = pad_to_multiple(img, img, 32, 255, 255)
    x = torch.from_numpy(pi)[None, None].float().to(device) / 255
    return predict_safe(model, x, amp)[: img.shape[0], : img.shape[1]].astype(np.uint8)


def parse_v2(model, img, device, amp, no_text):
    """v2A: parse, erase the text found with that parse, parse again. Others: one plain parse."""
    pred = predict(model, img, device, amp)
    if not no_text:
        return pred
    m = text_mask(img, pred)
    return predict(model, erase(img, m, pred, wall_px=8), device, amp) if m.any() else pred


@torch.no_grad()
def evaluate(model, folder, plans, device, amp, no_text, cfg):
    """{'normal': metrics, 'text erased': metrics} on whole drawings (text erased with the GT labels)."""
    model.eval()
    vc = cfg["val"]
    res = {}
    for cond in ("normal", "text erased"):
        m = ParserMetrics(vc["component_classes"], vc["component_iou"], cfg["data"]["ignore_index"])
        for p in plans:
            img, lbl = read_pair(folder, p)
            if cond == "text erased":
                img = erase_text(img, lbl)
            m.update(parse_v2(model, img, device, amp, no_text), lbl)
        r = m.result()
        r["room_mIoU"] = float(np.mean([r[f"iou_{C16.names[c]}"] for c in ROOMS]))
        res[cond] = r
    model.train()
    return res


# --------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", choices=["A", "B"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--init", required=True, help="parser v1 checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--freq", default=None)
    ap.add_argument("--tuning", default=None, help="text file of tuning val plan ids (selection set)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tc, vc = cfg["train"], cfg["v2"]
    if args.smoke:
        cfg["data"]["crop"], tc["batch_size"], tc["workers"] = 128, 2, 0
    no_text = args.variant == "A"
    text_p = 1.0 if no_text else 0.5
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg | {"variant": args.variant}, sort_keys=False), encoding="utf-8")
    torch.manual_seed(tc["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = tc["amp"] and device.type == "cuda"

    model = Parser(cfg["model"], encoder_weights=None).to(device)
    ck = torch.load(args.init, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    enc = list(model.net.encoder.parameters())
    ids = {id(p) for p in enc}
    opt = torch.optim.AdamW([{"params": enc, "lr": tc["lr_encoder"]},
                             {"params": [p for p in model.parameters() if id(p) not in ids], "lr": tc["lr"]}],
                            weight_decay=tc["weight_decay"])
    base = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    crit = ParserLoss(class_weights(class_freq(args.freq), cfg["loss"]["ce_weight_cap"]),
                      cfg["loss"]["dice"], cfg["data"]["ignore_index"]).to(device)
    ds = TrainSet(Path(args.data) / "train", cfg, text_p)
    loader = torch.utils.data.DataLoader(ds, batch_size=tc["batch_size"], shuffle=True, num_workers=tc["workers"],
                                         pin_memory=device.type == "cuda", drop_last=True,
                                         persistent_workers=tc["workers"] > 0)
    val_dir = Path(args.data) / "val"
    plans = list_plans(val_dir)
    if args.tuning:
        keep = set(Path(args.tuning).read_text(encoding="utf-8").split())
        plans = [p for p in plans if p in keep]
    plans = plans[: (2 if args.smoke else vc["val_plans"])]

    step, best, hours_done = 0, -1.0, 0.0
    last = out / "checkpoints" / "last.pt"
    wfile = out / "constraint_weights.yaml"
    if last.exists():
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        step, best, hours_done = st["step"], st["best"], st["hours"]
        weights = st["weights"]
        print(f"resumed at step {step}", flush=True)
    else:
        weights, G = calibrate(model, loader, crit, device, 2 if args.smoke else vc["calib_batches"], amp)
        wfile.write_text(yaml.safe_dump({"weights": weights, "median_grad_and_batches": G}, sort_keys=False),
                         encoding="utf-8")
        print("constraint weights:", {k: f"{v:.3g}" for k, v in weights.items()}, flush=True)
        res = evaluate(model, val_dir, plans, device, amp, no_text, cfg)          # v1 on the same set
        append_csv(out / "val_log.csv", val_row(0, 0.0, res))
        print("VAL v1 (step 0):", summary(res), flush=True)

    budget = tc["time_limit_hours"]
    t0 = time.time()

    def hours():
        return hours_done + (time.time() - t0) / 3600

    def save(path):
        st = {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step,
              "best": best, "hours": hours(), "weights": weights, "config": cfg, "variant": args.variant}
        torch.save(st, str(path) + ".tmp")
        os.replace(str(path) + ".tmp", path)

    last_val = last_ck = time.time()
    losses, t_log, n_log = [], time.time(), 0
    for x, y, s in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        f = lr_factor(step, hours() / budget, tc["warmup_steps"])
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * f
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
        rec, parts = crit(logits, y)
        terms = constraint_terms(logits, y, s)
        loss = rec + sum(weights.get(n, 0.0) * v for n, v in terms.items())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        scaler.step(opt)
        scaler.update()
        step += 1
        losses.append(float(loss))
        n_log += len(x)
        if step % (1 if args.smoke else tc["log_every"]) == 0:
            row = {"step": step, "hours": round(hours(), 4), "loss": round(float(np.mean(losses)), 5),
                   "ce": round(parts["ce"], 5), "dice": round(parts["dice"], 5),
                   **{n: round(float(v), 6) for n, v in terms.items()},
                   "samples_per_s": round(n_log / (time.time() - t_log), 2)}
            append_csv(out / "train_log.csv", row)
            print(row, flush=True)
            losses, t_log, n_log = [], time.time(), 0
        stop = (time.time() - t_start) / 3600 >= budget or (args.smoke and step >= 4)
        if stop or time.time() - last_val >= tc["val_minutes"] * 60 or (args.smoke and step % 2 == 0):
            res = evaluate(model, val_dir, plans, device, amp, no_text, cfg)
            append_csv(out / "val_log.csv", val_row(step, hours(), res))
            score = res["text erased"]["mIoU"] if no_text else 0.5 * (res["normal"]["mIoU"] + res["text erased"]["mIoU"])
            print(f"VAL step {step}: {summary(res)} score {score:.4f}", flush=True)
            if score > best:
                best = score
                save(out / "checkpoints" / "best.pt")
            last_val = time.time()
        if stop or time.time() - last_ck >= tc["checkpoint_minutes"] * 60:
            save(last)
            last_ck = time.time()
        if stop:
            break
    print(f"stopped at step {step}, {hours():.2f} h, best score {best:.4f}", flush=True)


def val_row(step, h, res):
    row = {"step": step, "hours": round(h, 4)}
    for cond, r in res.items():
        k = "txt_" if cond == "text erased" else ""
        row |= {f"{k}mIoU": round(r["mIoU"], 5), f"{k}room_mIoU": round(r["room_mIoU"], 5),
                f"{k}iou_wall": round(r["iou_wall"], 5), f"{k}F1_door": round(r["F1_door"], 5),
                f"{k}F1_window": round(r["F1_window"], 5)}
    return row


def summary(res):
    return " | ".join(f"{c}: mIoU {r['mIoU']:.4f} rooms {r['room_mIoU']:.4f}" for c, r in res.items())


if __name__ == "__main__":
    main()
