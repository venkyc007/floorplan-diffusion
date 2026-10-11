"""Full val-benchmark evaluation (all 7,772 cases) of every model so far.

Settings:
  nearest_fill                   baseline (CPU)
  unet                           U-Net baseline (best.pt)
  run1_ddim50                    diffusion run 1, stochastic DDIM 50 (eta 1)
  run1_refine600                 diffusion run 1, refine from one-shot at t = 600
  ft_constraints_refine600       constraint fine-tune, refine from one-shot at t = 600
  ft_control_refine600           control fine-tune, refine from one-shot at t = 600

Step 1 (one process per GPU) samples its settings and saves the predicted
label maps (uint8). Step 2 scores every setting with all metrics (overall +
by type / location / position / size bin), in parallel on the CPU cores.

    python -m src.full_eval --part 0 --parts 2 --preds /tmp/preds --ckpt-run1 R1 --ckpt-unet U ...
    python -m src.full_eval --score --preds /tmp/preds --out DIR
    python -m src.full_eval --figure --preds unused --out outputs --ckpt-...   # 8 large-hole cases, local
Local code check: add --limit 4 --quick
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.corruptions import to_model_input
from src.diagnose_diffusion import one_shot, sample
from src.eval.baselines import SHOW, nearest_fill
from src.eval.metrics import evaluate_benchmark, load_benchmark, to_markdown
from src.evaluate import load_trained
from src.sampling_experiment import refine
from src.utils import project_path

# setting: (checkpoint argument, method)
SETTINGS = {
    "unet": ("unet", "unet"),
    "unet_matched": ("unet_matched", "unet"),
    "run1_ddim50": ("run1", "ddim50"),
    "run1_refine600": ("run1", "refine600"),
    "ft_constraints_refine600": ("ft_constraints", "refine600"),
    "ft_control_refine600": ("ft_control", "refine600"),
    # refine-level sweep of the constraint fine-tune (Day 6, B.3)
    "ft_constraints_refine200": ("ft_constraints", "refine200"),
    "ft_constraints_refine400": ("ft_constraints", "refine400"),
    "ft_constraints_refine800": ("ft_constraints", "refine800"),
    "ft_constraints_ddim50": ("ft_constraints", "ddim50"),
}
# Balanced over 2 GPUs (DDIM 50 ~ 0.5 s/case, refine-600 ~ 0.33 s/case, U-Net ~ 0.02 s/case).
PARTS = {0: ["run1_ddim50", "run1_refine600"],
         1: ["unet", "ft_constraints_refine600", "ft_control_refine600"]}
ORDER = ["nearest_fill", "unet", "unet_matched", "run1_ddim50", "run1_refine600",
         "ft_constraints_refine600", "ft_control_refine600"]


def benchmark(limit=None):
    bench, meta = load_benchmark(project_path("data/benchmarks/val_benchmark.npz"))
    if limit:
        bench, meta = {k: v[:limit] for k, v in bench.items()}, meta.iloc[:limit]
    return bench, meta


@torch.no_grad()
def predict_all(model, method, bench, device, steps, seed, bs=64):
    """Label maps for every case; the 7-channel inputs are built batch by batch."""
    gen = torch.Generator(device=device).manual_seed(seed)
    out = []
    for s in range(0, len(bench["masks"]), bs):
        cond = torch.from_numpy(np.stack([to_model_input(c, m) for c, m in
                                          zip(bench["inputs"][s:s + bs], bench["masks"][s:s + bs])])).to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            if method == "unet":
                pred = model.predict(cond)
            elif method == "ddim50":
                pred = sample(model, cond, "ddim", steps, 1.0, True, gen)[0]
            else:  # refineT: one-shot, then refine from t = T (e.g. refine600)
                start = one_shot(model, cond, False, gen)
                pred = refine(model, cond, start, int(method[len("refine"):]), steps, gen)
        out.append(pred.cpu().numpy().astype(np.uint8))
    return np.concatenate(out)


def run_part(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bench, _ = benchmark(args.limit)
    ckpts = {"unet": args.ckpt_unet, "run1": args.ckpt_run1, "unet_matched": args.ckpt_unet_matched,
             "ft_constraints": args.ckpt_ft_constraints, "ft_control": args.ckpt_ft_control}
    preds = Path(args.preds)
    preds.mkdir(parents=True, exist_ok=True)
    timing = {}
    names = args.only.split(",") if args.only else (PARTS[args.part] if args.parts > 1 else sum(PARTS.values(), []))
    for k, name in enumerate(names):
        ckpt, method = SETTINGS[name]
        model, _ = load_trained(ckpts[ckpt], device)
        t0 = time.time()
        labels = predict_all(model, method, bench, device, 3 if args.quick else 50, seed=100 + k)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timing[name] = round((time.time() - t0) / len(labels), 4)
        np.save(preds / f"{name}.npy", labels)
        if args.keep_preds:   # compressed per-case predictions for later metrics (no GPU needed)
            Path(args.keep_preds).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(Path(args.keep_preds) / f"{name}.npz", labels=labels, plan_ids=bench["plan_ids"])
        print(f"{name}: {len(labels)} cases, {timing[name]} s/case", flush=True)
    json.dump(timing, open(preds / f"timing_part{args.part}.json", "w"))


def score_one(job):
    """Score one setting (runs in a worker process)."""
    name, preds_dir, limit = job
    bench, meta = benchmark(limit)
    if name == "nearest_fill":
        fn = nearest_fill
    else:
        labels = np.load(Path(preds_dir) / f"{name}.npy")
        state = {"i": 0}

        def fn(batch):
            n = len(batch["masks"])
            out = labels[state["i"]:state["i"] + n]
            state["i"] += n
            return out
    table, _ = evaluate_benchmark(fn, bench, meta, batch_size=256)
    return name, table


def score(args):
    preds = Path(args.preds)
    timing = {}
    for p in preds.glob("timing_part*.json"):
        timing.update(json.load(open(p)))
    names = (args.only.split(",") if args.only else
             ["nearest_fill"] + [n for n in ORDER[1:] if (preds / f"{n}.npy").exists()])
    with Pool(min(len(names), os.cpu_count() or 1)) as pool:
        tables = dict(pool.map(score_one, [(n, str(preds), args.limit) for n in names]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    full = pd.concat({n: tables[n] for n in names}, names=["setting"])
    full.to_csv(out / "full_eval.csv")

    overall = pd.DataFrame({n: tables[n].loc["overall", SHOW] for n in names}).T
    overall["sec_per_case"] = [timing.get(n, float("nan")) for n in names]
    overall.index.name = "setting"
    lines = [f"# Full val-benchmark evaluation ({int(overall['n'].iloc[0])} cases)", "",
             "## Overall", "", to_markdown(overall), ""]
    key = ["tmIoU", "mIoU", "F1_door", "F1_window", "F1_room", "conn_err", "room_leak", "room_count_err"]
    for metric in key:
        by = pd.DataFrame({n: tables[n][metric] for n in names})
        by.index.name = "group"
        lines += [f"## {metric} by type / location / position / size", "", to_markdown(by), ""]
    (out / "full_eval.md").write_text("\n".join(lines), encoding="utf-8")
    print(to_markdown(overall[["n", "tmIoU", "mIoU", "F1_door", "F1_window", "F1_room",
                               "conn_err", "room_leak", "room_count_err", "sec_per_case"]]))


def figure(args, n=8, seed=7):
    """Re-sample n large-hole val cases with every method and draw them side by side.

    Runs locally (CPU is fine for 8 cases). The diffusion samples are NEW draws
    from the same checkpoints, not the exact predictions that were scored."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.eval.figures import colorize

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bench, meta = benchmark()
    large = meta.index[meta.size_bin == "large"].to_numpy()
    pick = np.sort(np.random.default_rng(seed).choice(large, n, replace=False))
    sub = {k: v[pick] for k, v in bench.items()}
    ckpts = {"unet": args.ckpt_unet, "run1": args.ckpt_run1,
             "ft_constraints": args.ckpt_ft_constraints, "ft_control": args.ckpt_ft_control}
    preds = {"nearest_fill": nearest_fill({"corrupted": sub["inputs"], "masks": sub["masks"]})}
    for k, name in enumerate(ORDER[1:]):
        ckpt, method = SETTINGS[name]
        model, _ = load_trained(ckpts[ckpt], device)
        preds[name] = predict_all(model, method, sub, device, 3 if args.quick else 50, seed=200 + k)
        print("sampled", name, flush=True)
    titles = {"nearest_fill": "nearest fill", "unet": "U-Net", "run1_ddim50": "run 1, DDIM 50",
              "run1_refine600": "run 1, refine t=600", "ft_constraints_refine600": "fine-tune + constraints",
              "ft_control_refine600": "fine-tune control"}
    cols = ["input"] + ORDER + ["ground truth"]
    fig, axes = plt.subplots(n, len(cols), figsize=(2.2 * len(cols), 2.3 * n))
    for r in range(n):
        m = meta.loc[pick[r]]
        for c, col in enumerate(cols):
            if col == "input":
                img, title = sub["inputs"][r], f"input ({m.type}, plan {m.plan_id})"
            elif col == "ground truth":
                img, title = sub["targets"][r], "ground truth"
            else:
                img, title = np.where(sub["masks"][r], preds[col][r], sub["targets"][r]), titles[col]
            axes[r, c].imshow(colorize(img.astype(np.uint8)), interpolation="nearest")
            axes[r, c].set_title(title, fontsize=7)
            axes[r, c].axis("off")
    fig.suptitle("8 large-hole val cases (diffusion columns: fresh samples from the evaluated checkpoints)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))          # leave room for the title
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "full_eval_large_holes.png", dpi=85)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--preds", required=True, help="folder for predicted label maps")
    parser.add_argument("--out", help="folder for the result tables (--score)")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--figure", action="store_true", help="draw 8 large-hole cases (local)")
    parser.add_argument("--ckpt-run1")
    parser.add_argument("--ckpt-unet")
    parser.add_argument("--ckpt-unet-matched")
    parser.add_argument("--only", help="comma-separated settings to run/score (default: all)")
    parser.add_argument("--keep-preds", help="also save compressed per-case predictions (.npz) here")
    parser.add_argument("--ckpt-ft-constraints")
    parser.add_argument("--ckpt-ft-control")
    parser.add_argument("--limit", type=int, help="only the first N cases (code check)")
    parser.add_argument("--quick", action="store_true", help="3 sampling steps (code check)")
    args = parser.parse_args()
    if args.figure:
        figure(args)
    elif args.score:
        score(args)
    else:
        run_part(args)


if __name__ == "__main__":
    main()
