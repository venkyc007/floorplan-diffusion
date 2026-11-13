"""Novelty round 2 - STRUCTURE FIRST: one plan-level network that draws the missing walls.

    python -m src.novelty.structure --config configs/novelty/structure.yaml --train large9_train_*.zip \
        --val large9_tune_*.zip --out <run dir> [--no-ink] [--no-rays] [--seed 1] [--smoke]

Why (tuning drawings, round 1): the region locator misses one large omission in five, and the fill models, even
with the box given, bring back only about half of the missing wall pixels: they often give the two sides of a
missing wall different room types but leave the wall itself out, because its exact position is uncertain by a few
pixels and a per-pixel arg-max then prefers "room". Walls are what makes a missing room a room, so this network
is trained for walls only, with a target and a decision rule that tolerate a few pixels:

  input    the parser's 16-class map of the WHOLE plan (one-hot)
           + 2 channels of wall-continuation rays (src/novelty/rays.py: where visible walls would go if they continued)
           + 1 channel "ink": where the drawing has any ink (binarised, so paper tone and erasure smudges are
             invisible; a room that was never drawn has no label and no furniture)
  output   2 probability maps: "a missing wall runs here" (the erased walls, widened by 1 px on each side) and
           "this area is missing" (the locator's target, as a helper task)
  model    U-Net with an ImageNet ResNet-34 encoder (as the locator and detector v3), BCE + Dice per map
  data     src/novelty/data.StructureSet: 70% erased versions, 30% intact drawings (both targets all zero)
  decode   decode_walls: probability >= tau -> thinned to a 1 px centre line -> short specks dropped -> redrawn 3 px
           thick. So a wall the network places within a few pixels becomes one clean wall, not a blur or nothing.

The decoded walls are (a) the region proposal - where walls are added, something was missing - and (b) the
structure handed to the diffusion fill, which then adds doors, room types and fixtures (src/novelty/pipeline.py).
Validation (TUNING drawings, whole maps, decode at tau 0.5): share of omissions "recovered" (>= half of the truly
missing wall pixels back within 2 px, >= half of the added walls within 2 px of a true wall), mean recall and
precision, and the wall pixels added per INTACT drawing (false additions). best.pt by recovered.
"""

import argparse
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from skimage.morphology import skeletonize

from src.novelty.data import BG, StructureSet, arrays, dec, load_set, ray_bytes
from src.novelty.locate import append_csv, dice_loss
from src.stage2.data.classes16 import C16
from src.stage2.data.erasure import grown

WALL = C16.index["wall"]
N_EXTRA = 3                      # rays (free, junction) + ink


def build_structure(mcfg, weights="imagenet"):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=mcfg["encoder"], encoder_weights=weights, in_channels=C16.n + N_EXTRA, classes=2)


def load_structure(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_structure(ck["config"]["model"], None)
    model.load_state_dict(ck["model"])
    model.use = ck["config"]["model"].get("use", {"rays": True, "ink": True})
    return model.eval().to(device)


def features(lab, extra, use=None):
    """lab (B, H, W) int64, extra (B, 3, H, W) float [free rays, junction rays, ink] -> (B, 19, H, W).
    use = {"rays": bool, "ink": bool}: a switched-off input is zeroed (ablations keep the same network)."""
    extra = extra.clone()
    if use is not None and not use.get("rays", True):
        extra[:, :2] = 0
    if use is not None and not use.get("ink", True):
        extra[:, 2:] = 0
    return torch.cat([F.one_hot(lab, C16.n).permute(0, 3, 1, 2).float(), extra], 1)


def extra_maps(parsed, ink=None, rays=None):
    """(3, H, W) float32: free rays, junction rays, ink in [0, 1] (zeros where not available)."""
    if rays is None:
        from src.novelty.rays import continuation_rays
        free, junction = continuation_rays(parsed)
    else:
        free, junction = rays == 2, rays == 1
    ink = np.zeros(parsed.shape, np.float32) if ink is None else ink.astype(np.float32) / 255
    return np.stack([free.astype(np.float32), junction.astype(np.float32), ink])


@torch.no_grad()
def predict(models, parsed, ink=None, device="cpu", tta=False, rays=None):
    """(wall heat, area probability), both (H, W) in [0, 1]. models: one network or a list (mean).
    tta: also average over the 8 rotations / flips of the map."""
    models = models if isinstance(models, (list, tuple)) else [models]
    H, W = parsed.shape
    ph, pw = -(-max(H, 64) // 32) * 32, -(-max(W, 64) // 32) * 32
    lab = np.full((ph, pw), BG, np.uint8)
    lab[:H, :W] = parsed
    ex = np.zeros((N_EXTRA, ph, pw), np.float32)
    ex[:, :H, :W] = extra_maps(parsed, ink, rays)
    views = [(k, f) for f in (False, True) for k in range(4)] if tta else [(0, False)]
    acc = torch.zeros(2, ph, pw)
    for k, flip in views:
        l = np.rot90(lab[:, ::-1] if flip else lab, k)
        e = np.rot90(ex[:, :, ::-1] if flip else ex, k, axes=(1, 2))
        lt = torch.from_numpy(np.ascontiguousarray(l).astype(np.int64))[None].to(device)
        et = torch.from_numpy(np.ascontiguousarray(e))[None].to(device)
        p = sum(torch.sigmoid(m(features(lt, et, getattr(m, "use", None))).float())[0] for m in models) / len(models)
        p = torch.rot90(p, -k, dims=(1, 2))
        acc += (p.flip(2) if flip else p).cpu()
    acc /= len(views)
    return acc[0, :H, :W].numpy(), acc[1, :H, :W].numpy()


def decode_walls(heat, parsed, tau=0.5, min_len=6, thick=1):
    """Wall heat -> bool map of wall pixels to ADD: heat >= tau, thinned to its centre line, pieces shorter than
    min_len px dropped, redrawn (2 * thick + 1) px wide; pixels that are already wall in the parsed map excluded."""
    sk = skeletonize(heat >= tau)
    n, comp, st, _ = cv2.connectedComponentsWithStats(sk.astype(np.uint8), connectivity=8)
    keep = np.zeros(heat.shape, bool)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_len:
            keep |= comp == i
    return grown(keep, thick) & (parsed != WALL)


def wall_clusters(added, join=8, pad=6, min_px=12):
    """Boxes (y0, x0, y1, x1) around groups of added walls: pieces closer than `join` px form one group; groups
    with fewer than min_px added pixels are dropped; `pad` px of margin."""
    n, comp = cv2.connectedComponents(grown(added, join).astype(np.uint8), connectivity=8)
    out, (H, W) = [], added.shape
    for i in range(1, n):
        m = (comp == i) & added
        if m.sum() < min_px:
            continue
        ys, xs = np.nonzero(m)
        out.append((max(0, int(ys.min()) - pad), max(0, int(xs.min()) - pad), min(H - 1, int(ys.max()) + pad),
                    min(W - 1, int(xs.max()) + pad)))
    return out


def targets(gt, om):
    """(2, H, W) float32: the erased walls widened by 1 px, and the omission area."""
    band = grown((om == 2) & (gt == WALL), 1)
    return np.stack([band.astype(np.float32), (om > 0).astype(np.float32)])


@torch.no_grad()
def validate(model, samples, device, tau=0.5):
    from src.novelty.eval_maps import wall_recovery
    model.eval()
    rec, rc, pr, add_intact, flagged_intact = [], [], [], [], []
    for s in samples:
        parsed, gt, om = arrays(s)
        ink = dec(s["ink"]) if s.get("ink") is not None else None
        heat, _ = predict(model, parsed, ink, device, rays=dec(s["rays"]) if s.get("rays") is not None else None)
        added = decode_walls(heat, parsed, tau)
        plan = np.where(added, WALL, parsed).astype(np.uint8)
        if s["version"] == 0:
            add_intact.append(int(added.sum()))
            flagged_intact.append(len(wall_clusters(added)) > 0)
            continue
        ys, xs = np.nonzero(om > 0)
        region = np.zeros(parsed.shape, bool)
        region[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
        w = wall_recovery(plan, parsed, gt, om == 2, region)
        rec.append(w["recovered"])
        rc.append(w["miss_recall"] if w["miss_recall"] != "" else np.nan)
        pr.append(w["add_precision"] if w["add_precision"] != "" else np.nan)
    model.train()
    return {"recovered": float(np.mean(rec)), "miss_recall": float(np.nanmean(rc)), "add_precision": float(np.nanmean(pr)),
            "added_px_per_intact": float(np.mean(add_intact)) if add_intact else 0.0,
            "intact_flagged": float(np.mean(flagged_intact)) if flagged_intact else 0.0}


def _rays_job(b):
    return ray_bytes(dec(b))


def add_rays(samples, procs=4):
    """Precompute the wall-continuation rays of every sample's parsed map (s["rays"], PNG bytes: 1 junction, 2 free)."""
    t = time.time()
    if procs > 1:
        import multiprocessing as mp
        with mp.Pool(procs) as pool:
            out = pool.map(_rays_job, [s["map"] for s in samples], chunksize=64)
    else:
        out = [_rays_job(s["map"]) for s in samples]
    for s, r in zip(samples, out):
        s["rays"] = r
    print(f"rays for {len(samples)} maps in {time.time() - t:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-ink", action="store_true", help="ablation: the ink map is not given")
    ap.add_argument("--no-rays", action="store_true", help="ablation: the continuation rays are not given")
    ap.add_argument("--max-version", type=int, default=None, help="train on versions <= this only (3 = the first set)")
    ap.add_argument("--hours", type=float, default=None, help="override train.time_limit_hours")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tc, dc = cfg["train"], cfg["data"]
    cfg["model"]["use"] = {"rays": not args.no_rays, "ink": not args.no_ink}
    use = cfg["model"]["use"]
    if args.seed is not None:
        tc["seed"] = args.seed
    if args.hours:
        tc["time_limit_hours"] = args.hours
    if args.smoke:
        tc = tc | {"batch_size": 2, "warmup_steps": 2, "log_every": 1}
        dc = dc | {"num_workers": 0, "canvas": 128, "ray_procs": 1}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    torch.manual_seed(tc["seed"])
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    samples = load_set(args.train, limit=40 if args.smoke else None, max_version=args.max_version)
    vsamples = load_set(args.val, limit=12 if args.smoke else None)
    add_rays(samples, dc.get("ray_procs", 4))
    add_rays(vsamples, dc.get("ray_procs", 4))
    loader = torch.utils.data.DataLoader(StructureSet(samples, dc["canvas"], dc["erased_frac"]), batch_size=tc["batch_size"],
                                         num_workers=dc["num_workers"], drop_last=True, pin_memory=amp,
                                         persistent_workers=dc["num_workers"] > 0)
    model = build_structure(cfg["model"], None if args.smoke else "imagenet").to(device)
    model.use = use
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step, best, budget = 0, -9.0, tc["time_limit_hours"]
    hours = lambda: (time.time() - t_start) / 3600
    print(f"structure net: {len(samples)} train versions ({sum(s['version'] > 0 for s in samples)} erased, "
          f"{sum(s.get('ink') is not None for s in samples)} with ink), {len(vsamples)} val versions, inputs {use}, "
          f"device {device}", flush=True)
    last_val, t_log, losses = time.time(), time.time(), []
    for lab, extra, tgt in loader:
        x, y = features(lab.to(device), extra.to(device), use), tgt.to(device)
        f = (step + 1) / tc["warmup_steps"] if step < tc["warmup_steps"] else 0.5 * (1 + math.cos(math.pi * min(hours() / budget, 1)))
        for g in opt.param_groups:
            g["lr"] = tc["lr"] * f
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
        lg = logits.float()
        loss = (tc["wall_weight"] * (F.binary_cross_entropy_with_logits(lg[:, :1], y[:, :1]) + dice_loss(lg[:, :1], y[:, :1]))
                + F.binary_cross_entropy_with_logits(lg[:, 1:], y[:, 1:]) + dice_loss(lg[:, 1:], y[:, 1:]))
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
            print(f"VAL step {step}: " + " ".join(f"{k} {v:.3f}" for k, v in res.items()), flush=True)
            append_csv(out / "val_log.csv", {"step": step, "hours": round(hours(), 4), **{k: round(v, 5) for k, v in res.items()}})
            state = {"model": model.state_dict(), "step": step, "config": cfg, "val": res}
            if res["recovered"] > best:
                best = res["recovered"]
                torch.save(state, out / "checkpoints" / "best.pt")
            torch.save(state, str(out / "checkpoints" / "last.pt") + ".tmp")
            os.replace(str(out / "checkpoints" / "last.pt") + ".tmp", out / "checkpoints" / "last.pt")
            last_val = time.time()
        if stop:
            break
    print(f"stopped at step {step}, {hours():.2f} h, best recovered {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
