"""Score saved full-val predictions with all metrics (Day 6, B.3 + B.4). CPU only.

Inputs: per-setting predictions saved by the Kaggle kernels
(<preds>/<setting>.npz with labels (N,128,128) uint8 + plan_ids, in val
benchmark order). Nearest fill is computed here.

    python -m src.stage1.arch_eval --preds outputs/kaggle/arch_eval/preds --out outputs

Steps:
  1. per case, for every setting: the Day 2 metrics (sample_stats) + the
     architectural metrics (arch_metrics.case_metrics), saved to
     outputs/stage1/arch_eval/per_case/<setting>.csv.gz (used by src/stage1/stats.py);
  2. post-processing: N in {4, 6, 8} tried on PP_PICK; the best N (highest
     architectural validity, tie -> tolerant mIoU) is applied to every
     setting ("<setting>+pp");
  3. outputs/stage1/arch_eval.md: overall table (all metrics, ground-truth reference
     rates) + breakdowns by type and size; outputs/stage1/refine_tradeoff.png;
     outputs/stage1/arch_violations.png (8 cases, one violation type each).
"""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.stage1.eval.arch_metrics import case_metrics, summarize_arch
from src.stage1.eval.baselines import nearest_fill
from src.stage1.eval.metrics import load_benchmark, sample_stats, summarize, to_markdown
from src.stage1.postprocess import clean
from src.utils import project_path

PP_PICK = "ft_constraints_refine600"
PP_N = [4, 6, 8]
PP_MAX_GT_CHANGED = 1.0   # % of REAL plans the clean-up may alter (it also removes legitimate stubs)
SWEEP = ["ft_constraints_refine200", "ft_constraints_refine400", "ft_constraints_refine600",
         "ft_constraints_refine800", "ft_constraints_ddim50"]
MAIN = ["nearest_fill", "unet", "unet_matched", "run1_refine600", "ft_constraints_refine600", "ft_control_refine600"]
OLD_KEYS = ["tmIoU", "mIoU", "pixel_acc", "F1_door", "F1_window", "F1_room", "conn_err",
            "open_on_wall", "room_leak", "room_count_err", "alignment"]


def load_preds(preds_dir, name, bench):
    if name == "nearest_fill":
        return nearest_fill({"corrupted": bench["inputs"], "masks": bench["masks"]})
    z = np.load(Path(preds_dir) / f"{name}.npz")
    n = len(z["plan_ids"])                                    # a code-check run may hold only the first n cases
    assert (z["plan_ids"] == bench["plan_ids"][:n]).all(), f"{name}: case order differs from the benchmark"
    return z["labels"]


def score_setting(job):
    """Per-case old + architectural metrics of one setting (worker process)."""
    name, preds_dir, pp_n, limit, *split = job          # optional 5th item: "val" (default) or "test"
    bench, meta = load_benchmark(project_path(f"data/benchmarks/{split[0] if split else 'val'}_benchmark.npz"))
    th = yaml.safe_load(open(project_path("configs/stage1/arch_metrics.yaml"), encoding="utf-8"))
    base = name.split("+pp")[0]
    preds = load_preds(preds_dir, base, bench)
    n = len(preds) if limit is None else limit
    rows = []
    for i in range(n):
        pred, gt, mask = preds[i], bench["targets"][i], bench["masks"][i]
        if pp_n:
            pred = clean(pred, mask, pp_n)
        rows.append({**sample_stats(pred, gt, mask), **case_metrics(pred, gt, mask, th)})
    per_case = pd.concat([meta.iloc[:n].reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    return name, per_case


def gt_changed_pct(job):
    """% of ground-truth plans that post-processing with N would alter (worker)."""
    n, limit = job
    bench, _ = load_benchmark(project_path("data/benchmarks/val_benchmark.npz"))
    m = len(bench["masks"]) if limit is None else limit
    changed = sum(bool((clean(bench["targets"][i], bench["masks"][i], n) != bench["targets"][i]).any())
                  for i in range(m))
    return n, 100 * changed / m


def run_jobs(jobs, workers):
    with Pool(workers) as pool:
        return dict(pool.map(score_setting, jobs))


def summary_row(per_case):
    return {**{k: summarize(per_case)[k] for k in OLD_KEYS}, **summarize_arch(per_case.to_dict("records"))}


def tables(results, names):
    overall = pd.DataFrame({n: summary_row(results[n]) for n in names}).T
    overall.index.name = "setting"
    groups = {}
    for col in ("type", "size_bin"):
        for value in sorted(results[names[0]][col].unique()):
            groups[f"{col}={value}"] = pd.DataFrame(
                {n: summary_row(results[n][results[n][col] == value]) for n in names}).T
    return overall, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out", default="outputs/stage1")
    parser.add_argument("--limit", type=int, help="first N cases only (code check)")
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    args = parser.parse_args()
    out = project_path(args.out)
    per_case_dir = out / "arch_eval" / "per_case"
    per_case_dir.mkdir(parents=True, exist_ok=True)
    available = ["nearest_fill"] + sorted(p.stem for p in Path(args.preds).glob("*.npz"))

    # 1. every setting as sampled
    results = run_jobs([(n, args.preds, None, args.limit) for n in available], args.workers)
    # 2. pick N for post-processing on PP_PICK, then apply it everywhere
    trial = run_jobs([(f"{PP_PICK}+pp{n}", args.preds, n, args.limit) for n in PP_N], args.workers)
    pick = pd.DataFrame({k: summary_row(v) for k, v in trial.items()}).T
    with Pool(min(len(PP_N), args.workers)) as pool:
        damage = dict(pool.map(gt_changed_pct, [(n, args.limit) for n in PP_N]))
    pick["gt_plans_changed_pct"] = [damage[int(k.split("+pp")[1])] for k in pick.index]
    allowed = pick[pick.gt_plans_changed_pct <= PP_MAX_GT_CHANGED]
    best_key = (allowed if len(allowed) else pick).sort_values(["arch_validity_pct", "tmIoU"], ascending=False).index[0]
    best_n = int(best_key.split("+pp")[1])
    pp = run_jobs([(f"{n}+pp", args.preds, best_n, args.limit) for n in available if n in MAIN],
                  args.workers)
    results.update(pp)
    for name, df in results.items():
        df.to_csv(per_case_dir / f"{name.replace('+', '_')}.csv.gz", index=False)

    names = [n for n in MAIN if n in results] + [f"{n}+pp" for n in MAIN if f"{n}+pp" in results]
    overall, groups = tables(results, names)
    sweep_names = [n for n in SWEEP if n in results]
    sweep, _ = tables(results, sweep_names) if sweep_names else (None, None)
    write_report(out, overall, groups, pick, best_n, sweep, len(results[names[0]]))
    if sweep is not None:
        plot_tradeoff(out, sweep, overall)
    plot_violations(out, results, args.preds)
    print(to_markdown(overall[["tmIoU", "conn_err", "room_leak", "room_count_err", "dead_ends_per_case",
                               "fragments_per_case", "opening_fit_pct", "arch_validity_pct"]]))


def write_report(out, overall, groups, pick, best_n, sweep, n_cases):
    arch_cols = ["dead_ends_per_case", "gt_dead_ends_per_case", "fragments_per_case", "gt_fragments_per_case",
                 "opening_bridge_per_case", "gt_opening_bridge_per_case", "opening_fit_pct", "gt_opening_fit_pct",
                 "bridges_pct", "thickness_pct", "aligned_pct", "door_spaces_per_case", "gt_door_spaces_per_case",
                 "window_out_per_case", "gt_window_out_per_case", "unreachable_per_case", "gt_unreachable_per_case",
                 "slivers_per_case", "gt_slivers_per_case", "rectangularity", "gt_rectangularity",
                 "arch_validity_pct"]
    lines = [f"# Architectural evaluation (full val benchmark, {n_cases} cases)", "",
             "Day 2 metrics + architectural validity metrics (`src/stage1/eval/arch_metrics.py`); every "
             "architectural rate is shown next to the ground truth's own rate on the same cases (`gt_` columns). "
             "`+pp` = with post-processing (`src/stage1/postprocess.py`). Per-case results: `arch_eval/per_case/`.", "",
             f"## Post-processing: choice of N (on {PP_PICK})", "",
             to_markdown(pick[["tmIoU", "conn_err", "dead_ends_per_case", "arch_validity_pct", "gt_plans_changed_pct"]]), "",
             f"Chosen N = **{best_n}**: highest architectural validity among the N that alter at most "
             f"{PP_MAX_GT_CHANGED}% of REAL plans (the clean-up also removes legitimate short stubs); "
             "ties broken by tolerant mIoU.", "",
             "## Overall: Day 2 metrics", "", to_markdown(overall[OLD_KEYS]), "",
             "## Overall: architectural metrics (prediction vs ground truth)", "",
             to_markdown(overall[arch_cols]), ""]
    key = ["tmIoU", "conn_err", "room_leak", "room_count_err", "dead_ends_per_case", "arch_validity_pct"]
    for g, t in groups.items():
        t.index.name = "setting"
        lines += [f"## {g}", "", to_markdown(t[key]), ""]
    if sweep is not None:
        sweep.index.name = "setting"
        lines += ["## Refine-level sweep (constraint fine-tune)", "",
                  to_markdown(sweep[["tmIoU", "mIoU", "conn_err", "room_leak", "room_count_err",
                                     "dead_ends_per_case", "arch_validity_pct"]]), "",
                  "Figure: `refine_tradeoff.png`.", ""]
    (out / "arch_eval.md").write_text("\n".join(lines), encoding="utf-8")


def plot_tradeoff(out, sweep, overall):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"ft_constraints_refine200": "refine t=200", "ft_constraints_refine400": "refine t=400",
              "ft_constraints_refine600": "refine t=600", "ft_constraints_refine800": "refine t=800",
              "ft_constraints_ddim50": "DDIM 50 (t=1000)"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, ycol, ylab in [(axes[0], "conn_err", "connectivity error (lower = better)"),
                           (axes[1], "arch_validity_pct", "architectural validity % (higher = better)")]:
        ax.plot(sweep["tmIoU"], sweep[ycol], "o-", color="tab:blue", label="constraint fine-tune")
        for name, row in sweep.iterrows():
            ax.annotate(labels.get(name, name), (row["tmIoU"], row[ycol]), fontsize=8,
                        textcoords="offset points", xytext=(4, 4))
        for ref, marker in [("unet", "s"), ("unet_matched", "D"), ("ft_control_refine600", "^")]:
            if ref in overall.index:
                ax.plot(overall.loc[ref, "tmIoU"], overall.loc[ref, ycol], marker, label=ref)
        ax.set(xlabel="tolerant mIoU (higher = better)", ylabel=ylab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Accuracy vs structure: how far to re-noise the one-shot guess (full val benchmark)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "refine_tradeoff.png", dpi=100)


VIOLATION_ORDER = ["dead_ends", "fragments", "opening_bridge", "door_spaces", "window_out",
                   "unreachable", "slivers", "low_rectangularity"]


def plot_violations(out, results, preds_dir, sources=("unet", "ft_constraints_refine600", "run1_refine600")):
    """8 cases, one per violation type: input | prediction (violation outlined) | ground truth.
    Each case is one where the prediction breaks the rule but the ground truth doesn't."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.stage1.eval.arch_metrics import MARGIN, grow, violation_locations
    from src.stage1.eval.figures import colorize
    bench, meta = load_benchmark(project_path("data/benchmarks/val_benchmark.npz"))
    th = yaml.safe_load(open(project_path("configs/stage1/arch_metrics.yaml"), encoding="utf-8"))
    picks, used = [], set()                              # a different case in every row
    for rule in VIOLATION_ORDER:
        for src in sources:
            if src not in results:
                continue
            df = results[src]
            col = f"pred_{rule}" if f"pred_{rule}" in df else None
            cands = df.index[df[col] > df[f"gt_{rule}"]] if col else df.index
            preds = load_preds(preds_dir, src, bench)
            found = None
            for i in [c for c in cands if c not in used][:400]:
                region = grow(bench["masks"][i], MARGIN)
                pred = np.where(bench["masks"][i], preds[i], bench["targets"][i])
                spots_p = violation_locations(pred, region, th)[rule]
                spots_g = violation_locations(bench["targets"][i], region, th)[rule]
                if len(spots_p) > len(spots_g):
                    found = (rule, src, i, pred, spots_p)
                    break
            if found:
                picks.append(found)
                used.add(found[2])
                break
    fig, axes = plt.subplots(len(picks), 3, figsize=(8, 2.7 * len(picks)), squeeze=False)
    for r, (rule, src, i, pred, spots) in enumerate(picks):
        outline = np.zeros(pred.shape, bool)
        for sp in spots:
            outline |= grow(sp, 2) & ~sp
        img = colorize(pred.astype(np.uint8)).copy()
        img[outline] = (255, 0, 255)                     # magenta outline around each violation
        panels = [(colorize(bench["inputs"][i]), f"input ({meta.type[i]}, plan {meta.plan_id[i]})"),
                  (img, f"{src}: {rule} ({len(spots)})"),
                  (colorize(bench["targets"][i]), "ground truth")]
        for c, (im, title) in enumerate(panels):
            axes[r, c].imshow(im, interpolation="nearest")
            axes[r, c].set_title(title, fontsize=8)
            axes[r, c].axis("off")
    fig.suptitle("Architectural violations (magenta outline), one type per row", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "arch_violations.png", dpi=110)


if __name__ == "__main__":
    main()
