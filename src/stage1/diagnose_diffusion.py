"""Diagnose the trained diffusion model (run 1) on the 256 fixed val cases.

Why: the diffusion model scored below the one-pass U-Net (tolerant mIoU
0.698 vs 0.794). This checks whether the cause is SAMPLING (steps, noise,
known-region reset, final output) or the DENOISER itself.

Settings (each scored with the Day 2 metrics + a speckle count):
  ddim20 / ddim50 / ddim100     DDIM, eta = 0 (deterministic)
  ddim50_eta1                   DDIM, eta = 1 (stochastic)
  ddpm250                       DDPM ancestral sampling, 250 steps
  ddim50_noreset                DDIM 50 without overwriting the known region
  oneshot_T                     one forward pass at t = T from pure noise
  oneshot_T_known               same, known region (clean) filled in outside the mask
  denoise_t<k>                  ground truth noised to t = k, one forward pass
For sampling settings both outputs are scored: argmax of the final x_t
("final") and argmax of the last predicted x0 ("x0").

Run on Kaggle by kaggle/kernels/stage1/diagnostic/run.py (one process per GPU):
    python -m src.stage1.diagnose_diffusion --diffusion D/best.pt --unet U/best.pt --part 0 --parts 2 --out DIR
    python -m src.stage1.diagnose_diffusion --report --out DIR
Local code check (tiny, CPU):  add --cases 2 --quick
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from diffusers import DDIMScheduler, DDPMScheduler

from src.stage1.data.classes import NUM_CLASSES
from src.stage1.data.corruptions import to_model_input
from src.stage1.eval.figures import colorize
from src.stage1.eval.metrics import load_benchmark, sample_stats, summarize, to_markdown
from src.stage1.evaluate import load_trained
from src.stage1.train import VAL_SEED
from src.utils import project_path

SAMPLERS = {   # name: (scheduler, steps, eta, reset known region)
    "ddim20": ("ddim", 20, 0.0, True), "ddim50": ("ddim", 50, 0.0, True),
    "ddim100": ("ddim", 100, 0.0, True), "ddim50_eta1": ("ddim", 50, 1.0, True),
    "ddpm250": ("ddpm", 250, None, True), "ddim50_noreset": ("ddim", 50, 0.0, False),
}
DENOISE_T = [50, 200, 400, 600, 800, 999]          # 999 = the last timestep ("t = 1000")
# Balanced split over 2 GPUs (about 270 network evaluations per case each).
PARTS = {0: ["ddpm250", "ddim20", "oneshot", "denoise", "unet"],
         1: ["ddim50", "ddim100", "ddim50_eta1", "ddim50_noreset"]}
SPECKLE_PX = 4
KEEP = ["tmIoU", "mIoU", "F1_door", "F1_window", "F1_room", "conn_err"]


# ----------------------------------------------------------------- data

def val_cases(n=256):
    """The same fixed val cases as training validation (src/stage1/train.py:val_subset)."""
    bench, meta = load_benchmark(project_path("data/benchmarks/val_benchmark.npz"))
    idx = np.sort(np.random.default_rng(VAL_SEED).choice(len(meta), n, replace=False))
    return bench, meta, idx


def figure_cases(meta, idx):
    """6 of the 256 cases for the figure: 3 large holes + door, wall, room."""
    m = meta.iloc[idx].reset_index(drop=True)
    picks = list(m.index[m.size_bin == "large"][[0, 5, 10]])
    for t in ("door", "wall", "room"):
        picks.append(int(m.index[(m.type == t) & ~m.index.isin(picks)][2]))
    return picks


def speckles(pred, gt, mask):
    """Tiny pieces (< SPECKLE_PX px, any class, 4-connected) touching the mask."""
    full = np.where(mask, pred, gt).astype(np.uint8)
    n = 0
    for c in range(NUM_CLASSES):
        k, lab, stats, _ = cv2.connectedComponentsWithStats((full == c).astype(np.uint8), connectivity=4)
        for i in range(1, k):
            if stats[i, cv2.CC_STAT_AREA] < SPECKLE_PX and mask[lab == i].any():
                n += 1
    return n


def score(preds, targets, masks):
    s = summarize([sample_stats(p, t, m) for p, t, m in zip(preds, targets, masks)])
    out = {k: round(float(s[k]), 4) for k in KEEP}
    out["speckles_per_case"] = round(float(np.mean([speckles(p, t, m) for p, t, m in zip(preds, targets, masks)])), 2)
    return out


# ------------------------------------------------------------- sampling

def schedulers(mcfg):
    common = dict(num_train_timesteps=mcfg["num_train_timesteps"], beta_schedule=mcfg["beta_schedule"],
                  prediction_type="sample", clip_sample=True)
    return {"ddim": DDIMScheduler(**common, set_alpha_to_one=True), "ddpm": DDPMScheduler(**common)}


def net(model, x, cond, t):
    with torch.autocast(cond.device.type, enabled=cond.device.type == "cuda"):
        return model(x, cond, t.expand(len(x))).float()


@torch.no_grad()
def sample(model, cond, kind, steps, eta, reset, gen):
    """Returns (argmax of final x_t, argmax of the last predicted x0)."""
    sched = schedulers(model.mcfg)[kind]
    sched.set_timesteps(steps, device=cond.device)
    mask = cond[:, -1:].bool()
    known = cond[:, :NUM_CLASSES] * 2 - 1

    def noised(t):
        noise = torch.randn(known.shape, generator=gen, device=cond.device)
        return sched.add_noise(known, noise, t.reshape(1).expand(len(known)))

    ts = sched.timesteps
    x = torch.randn(known.shape, generator=gen, device=cond.device)
    if reset:
        x = torch.where(mask, x, noised(ts[0]))
    x0 = None
    for i, t in enumerate(ts):
        x0 = net(model, x, cond, t).clamp(-1, 1)
        kw = {"eta": eta} if kind == "ddim" else {}
        x = sched.step(x0, t, x, generator=gen, **kw).prev_sample
        if reset:
            x = torch.where(mask, x, noised(ts[i + 1]) if i + 1 < len(ts) else known)
    return x.argmax(1), x0.argmax(1)


@torch.no_grad()
def one_shot(model, cond, fill_known, gen):
    """One forward pass at t = T from pure noise (optionally clean known region outside the mask)."""
    T = torch.tensor(model.mcfg["num_train_timesteps"] - 1, device=cond.device)
    x = torch.randn((len(cond), NUM_CLASSES) + cond.shape[2:], generator=gen, device=cond.device)
    if fill_known:
        x = torch.where(cond[:, -1:].bool(), x, cond[:, :NUM_CLASSES] * 2 - 1)
    return net(model, x, cond, T).argmax(1)


@torch.no_grad()
def denoise(model, cond, target, t, gen):
    """Noise the ground truth to level t, one forward pass, argmax of predicted x0."""
    sched = schedulers(model.mcfg)["ddpm"]
    x0 = target * 2 - 1
    noise = torch.randn(x0.shape, generator=gen, device=cond.device)
    tt = torch.tensor([t], device=cond.device)
    return net(model, sched.add_noise(x0, noise, tt.expand(len(x0))), cond, tt).argmax(1)


def batched(fn, cond, *extra, bs=32):
    """Run fn over the cases in batches; returns numpy array(s) and seconds per case."""
    outs, t0 = [], time.time()
    for s in range(0, len(cond), bs):
        r = fn(cond[s:s + bs], *[e[s:s + bs] for e in extra])
        outs.append([v.cpu().numpy() for v in (r if isinstance(r, tuple) else (r,))])
    if cond.device.type == "cuda":
        torch.cuda.synchronize()
    secs = (time.time() - t0) / len(cond)
    return [np.concatenate(z) for z in zip(*outs)], secs


# ----------------------------------------------------------------- main

def run_part(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bench, meta, idx = val_cases(args.cases or 256)
    targets, masks = bench["targets"][idx], bench["masks"][idx]
    cond = torch.from_numpy(np.stack([to_model_input(bench["inputs"][i], bench["masks"][i]) for i in idx])).to(device)
    target = torch.from_numpy((np.arange(NUM_CLASSES)[None, :, None, None] == targets[:, None]).astype(np.float32)).to(device)
    fig_idx = [i for i in figure_cases(meta, idx) if i < len(idx)] if not args.cases else [0]
    model, _ = load_trained(args.diffusion, device)
    gen = lambda: torch.Generator(device=device).manual_seed(0)  # noqa: E731
    results, figs = [], {}

    def record(name, preds, secs, **info):
        results.append({"setting": name, **info, **score(preds, targets, masks), "sec_per_case": round(secs, 3)})
        figs[name] = preds[fig_idx]
        print(json.dumps(results[-1]), flush=True)

    for job in PARTS[args.part] if args.parts > 1 else sum(PARTS.values(), []):
        if job in SAMPLERS:
            kind, steps, eta, reset = SAMPLERS[job]
            steps = min(steps, 3) if args.quick else steps
            g = gen()
            (final, x0), secs = batched(lambda c: sample(model, c, kind, steps, eta, reset, g), cond)
            record(job + " [final x_t]", final, secs, steps=steps)
            record(job + " [last x0]", x0, secs, steps=steps)
        elif job == "oneshot":
            for fill in (False, True):
                g = gen()
                (p,), secs = batched(lambda c: one_shot(model, c, fill, g), cond)
                record("oneshot_T" + ("_known" if fill else ""), p, secs, steps=1)
        elif job == "denoise":
            for t in DENOISE_T[:2] if args.quick else DENOISE_T:
                g = gen()
                (p,), secs = batched(lambda c, y: denoise(model, c, y, t, g), cond, target)
                record(f"denoise_t{t}", p, secs, steps=1)
        elif job == "unet":
            unet, _ = load_trained(args.unet, device)
            (p,), secs = batched(lambda c: unet.predict(c), cond)
            record("U-Net (best.pt)", p, secs, steps=1)
            results.append({"setting": "ground truth", "speckles_per_case":
                            round(float(np.mean([speckles(t, t, m) for t, m in zip(targets, masks)])), 2)})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out / f"part{args.part}.json", "w"), indent=1)
    np.savez_compressed(out / f"part{args.part}_figure_preds.npz", _fig_idx=np.array(fig_idx), **figs)


def report(args):
    out = Path(args.out)
    rows = sum((json.load(open(p)) for p in sorted(out.glob("part*.json"))), [])
    df = pd.DataFrame(rows).set_index("setting")
    bench, meta, idx = val_cases()
    figs = {}
    for p in sorted(out.glob("part*_figure_preds.npz")):
        figs.update(dict(np.load(p)))
    fig_idx = list(figs.pop("_fig_idx"))
    samplers = df[df.index.str.contains(r"\[")]
    best = samplers["tmIoU"].idxmax()
    df.to_csv(out / "diagnostic.csv")
    (out / "diagnostic_table.md").write_text(to_markdown(df.drop(columns=["steps"], errors="ignore")), encoding="utf-8")

    cols = [("input", None), ("U-Net", "U-Net (best.pt)"), ("diffusion one-shot", "oneshot_T"),
            ("diffusion DDIM 50", "ddim50 [final x_t]"), (f"best: {best}", best), ("ground truth", None)]
    cols = [(t, k) for t, k in cols if k is None or k in figs]   # (code checks run fewer settings)
    fig, axes = plt.subplots(len(fig_idx), len(cols), figsize=(2.6 * len(cols), 2.7 * len(fig_idx)), squeeze=False)
    for r, k in enumerate(fig_idx):
        i, m = idx[k], meta.iloc[idx[k]]
        mask, gt = bench["masks"][i], bench["targets"][i]
        for c, (title, key) in enumerate(cols):
            img = bench["inputs"][i] if c == 0 else gt if key is None else np.where(mask, figs[key][r], gt)
            axes[r, c].imshow(colorize(img.astype(np.uint8)), interpolation="nearest")
            axes[r, c].set_title(title if c else f"input ({m.type}, {m.size_bin})", fontsize=8)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(out / "diffusion_run1_diagnostic.png", dpi=90)
    print(to_markdown(df.drop(columns=["steps"], errors="ignore")))
    print("best sampling setting:", best)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diffusion", help="diffusion best.pt")
    parser.add_argument("--unet", help="U-Net best.pt")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", action="store_true", help="merge the parts into tables + figure")
    parser.add_argument("--cases", type=int, help="only the first N of the 256 cases (code check)")
    parser.add_argument("--quick", action="store_true", help="at most 3 sampling steps (code check)")
    args = parser.parse_args()
    report(args) if args.report else run_part(args)


if __name__ == "__main__":
    main()
