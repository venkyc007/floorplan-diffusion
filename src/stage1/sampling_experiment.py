"""Sampling experiment for diffusion run 1 (follows src/stage1/diagnose_diffusion.py).

On the same 256 fixed val cases:
  refine_t200/400/600  start from the ONE-SHOT prediction (as a sharp one-hot
                       plan), noise it to t = 200/400/600, then finish with the
                       stochastic DDIM steps at or below that level (of 50)
  ddim50_eta1          one stochastic DDIM 50 sample (reference)
  medoid_of_4          4 samples; keep the one that agrees most with the other
                       three (mean pairwise tolerant mIoU inside the mask).
                       Does NOT look at the ground truth.
  vote_of_4            pixel-wise majority vote of the 4 samples (averaging
                       brings the blur back)
  ORACLE_best_of_4     the sample closest to the ground truth: an UPPER BOUND,
                       not a usable method (it needs the answer)
plus the U-Net and the one-shot prediction for comparison.

Run on Kaggle by kaggle/kernels/stage1/sampling/run.py (one process per GPU):
    python -m src.stage1.sampling_experiment --diffusion D/best.pt --unet U/best.pt --part 0 --parts 2 --out DIR
    python -m src.stage1.sampling_experiment --report --out DIR
Local code check: add --cases 2 --quick
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.stage1.data.classes import NUM_CLASSES
from src.stage1.data.corruptions import to_model_input
from src.stage1.diagnose_diffusion import figure_cases, net, one_shot, sample, schedulers, speckles, val_cases
from src.stage1.eval.figures import colorize
from src.stage1.eval.metrics import sample_stats, summarize, to_markdown
from src.stage1.evaluate import load_trained

N_SAMPLES = 4
REFINE_T = [200, 400, 600]
KEEP = ["tmIoU", "mIoU", "F1_door", "F1_window", "F1_room", "conn_err", "room_leak", "room_count_err"]


def score(preds, targets, masks):
    s = summarize([sample_stats(p, t, m) for p, t, m in zip(preds, targets, masks)])
    out = {k: round(float(s[k]), 4) for k in KEEP}
    out["speckles_per_case"] = round(float(np.mean([speckles(p, t, m) for p, t, m in zip(preds, targets, masks)])), 2)
    return out


@torch.no_grad()
def refine(model, cond, start_labels, t0, steps, gen, callback=None):
    """Noise a sharp starting plan to level t0, then denoise with stochastic DDIM.
    `callback(done, total)` (optional) is called after every denoising step, e.g. for a progress bar."""
    sched = schedulers(model.mcfg)["ddim"]
    sched.set_timesteps(steps, device=cond.device)
    ts = [t for t in sched.timesteps if t <= t0]
    mask = cond[:, -1:].bool()
    known = cond[:, :NUM_CLASSES] * 2 - 1
    start = torch.nn.functional.one_hot(start_labels, NUM_CLASSES).permute(0, 3, 1, 2).float() * 2 - 1
    start = torch.where(mask, start, known)

    def noised(x0, t):
        noise = torch.randn(x0.shape, generator=gen, device=cond.device)
        return sched.add_noise(x0, noise, t.reshape(1).expand(len(x0)))

    x = noised(start, ts[0])
    for i, t in enumerate(ts):
        x0 = net(model, x, cond, t).clamp(-1, 1)
        x = sched.step(x0, t, x, eta=1.0, generator=gen).prev_sample
        x = torch.where(mask, x, noised(known, ts[i + 1]) if i + 1 < len(ts) else known)
        if callback:
            callback(i + 1, len(ts))
    return x.argmax(1)


def batched(fn, cond, bs=32):
    """Run fn over the cases in batches; returns (labels, seconds per case)."""
    outs, t0 = [], time.time()
    for s in range(0, len(cond), bs):
        outs.append(fn(s, cond[s:s + bs]).cpu().numpy().astype(np.uint8))
    if cond.device.type == "cuda":
        torch.cuda.synchronize()
    return np.concatenate(outs), (time.time() - t0) / len(cond)


def run_part(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bench, meta, idx = val_cases(args.cases or 256)
    cond = torch.from_numpy(np.stack([to_model_input(bench["inputs"][i], bench["masks"][i]) for i in idx])).to(device)
    model, _ = load_trained(args.diffusion, device)
    steps = 3 if args.quick else 50
    labels, secs = {}, {}
    gen = torch.Generator(device=device).manual_seed(1000 + args.part)   # different noise per part
    if args.part == 0 or args.parts == 1:
        labels["oneshot"], secs["oneshot"] = batched(lambda s, c: one_shot(model, c, False, gen), cond)
        start = torch.from_numpy(labels["oneshot"]).long().to(device)
        for t0 in REFINE_T:
            labels[f"refine_t{t0}"], secs[f"refine_t{t0}"] = batched(
                lambda s, c: refine(model, c, start[s:s + len(c)], t0, steps, gen), cond)
    mine = range(N_SAMPLES) if args.parts == 1 else range(args.part * 2, args.part * 2 + 2)
    for k in mine:
        labels[f"sample{k}"], secs[f"sample{k}"] = batched(
            lambda s, c: sample(model, c, "ddim", steps, 1.0, True, gen)[0], cond)
    if args.part == args.parts - 1:
        unet, _ = load_trained(args.unet, device)
        labels["unet"], secs["unet"] = batched(lambda s, c: unet.predict(c), cond)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"part{args.part}_labels.npz", **labels)
    json.dump(secs, open(out / f"part{args.part}_seconds.json", "w"), indent=1)
    print("part", args.part, "done:", {k: round(v, 3) for k, v in secs.items()}, flush=True)


def pair_agreement(a, b, mask):
    """Tolerant mIoU of sample a against sample b inside the mask (no ground truth)."""
    return sample_stats(a, b, mask)["tmiou"]


def report(args):
    out = Path(args.out)
    labels, secs = {}, {}
    for p in sorted(out.glob("part*_labels.npz")):
        labels.update(dict(np.load(p)))
    for p in sorted(out.glob("part*_seconds.json")):
        secs.update(json.load(open(p)))
    n = len(labels["oneshot"])
    bench, meta, idx = val_cases(n)
    targets, masks = bench["targets"][idx], bench["masks"][idx]
    samples = np.stack([labels[f"sample{k}"] for k in range(N_SAMPLES)])        # (4, n, H, W)

    medoid, oracle, vote, picks = [], [], [], []
    for i in range(n):
        agree = np.zeros(N_SAMPLES)
        for a in range(N_SAMPLES):
            for b in range(N_SAMPLES):
                if a != b:
                    agree[a] += pair_agreement(samples[a, i], samples[b, i], masks[i])
        medoid.append(samples[agree.argmax(), i])
        picks.append(int(agree.argmax()))
        truth = [sample_stats(samples[k, i], targets[i], masks[i])["tmiou"] for k in range(N_SAMPLES)]
        oracle.append(samples[int(np.nanargmax(truth)), i])
        onehot = (np.arange(NUM_CLASSES)[:, None, None, None] == samples[:, i][None]).sum(1)
        vote.append(onehot.argmax(0).astype(np.uint8))
    sample_secs = sum(secs[f"sample{k}"] for k in range(N_SAMPLES))

    rows = [("U-Net baseline", labels["unet"], secs["unet"]),
            ("diffusion one-shot (t = T)", labels["oneshot"], secs["oneshot"]),
            ("DDIM 50, eta 1 (1 sample)", labels["sample0"], secs["sample0"])]
    rows += [(f"refine from one-shot, t = {t0}", labels[f"refine_t{t0}"], secs["oneshot"] + secs[f"refine_t{t0}"])
             for t0 in REFINE_T]
    rows += [("medoid of 4 samples (no ground truth)", np.stack(medoid), sample_secs),
             ("pixel vote of 4 samples (re-blurs)", np.stack(vote), sample_secs),
             ("ORACLE best of 4 vs ground truth (upper bound)", np.stack(oracle), sample_secs)]
    table = pd.DataFrame([{"setting": name, **score(p, targets, masks), "sec_per_case": round(s, 3)}
                          for name, p, s in rows]).set_index("setting")
    table.to_csv(out / "sampling_experiment.csv")
    (out / "sampling_table.md").write_text(to_markdown(table), encoding="utf-8")
    print(to_markdown(table))
    print("medoid picks per sample index:", np.bincount(picks, minlength=N_SAMPLES).tolist())

    best_refine = max(REFINE_T, key=lambda t: table.loc[f"refine from one-shot, t = {t}", "tmIoU"])
    cols = [("input", None), ("U-Net", labels["unet"]), ("one-shot", labels["oneshot"]),
            (f"refine t = {best_refine}", labels[f"refine_t{best_refine}"]),
            ("medoid of 4", np.stack(medoid)), ("ground truth", None)]
    fig_idx = [k for k in figure_cases(meta, idx) if k < n] if n == 256 else [0]
    fig, axes = plt.subplots(len(fig_idx), len(cols), figsize=(2.6 * len(cols), 2.7 * len(fig_idx)), squeeze=False)
    for r, k in enumerate(fig_idx):
        i, m = idx[k], meta.iloc[idx[k]]
        for c, (title, pred) in enumerate(cols):
            img = (bench["inputs"][i] if c == 0 else targets[k] if pred is None
                   else np.where(masks[k], pred[k], targets[k]))
            axes[r, c].imshow(colorize(img.astype(np.uint8)), interpolation="nearest")
            axes[r, c].set_title(title if c else f"input ({m.type}, {m.size_bin})", fontsize=8)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(out / "sampling_experiment.png", dpi=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diffusion")
    parser.add_argument("--unet")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--cases", type=int)
    parser.add_argument("--quick", action="store_true", help="3 sampling steps (code check)")
    args = parser.parse_args()
    report(args) if args.report else run_part(args)


if __name__ == "__main__":
    main()
