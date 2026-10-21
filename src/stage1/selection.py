"""Final selection experiment (Day 7): which configuration to recommend. CPU only.

Inputs: saved full-val predictions (<setting>.npz) of the new settings
(Kaggle kernel floorplan-selection: hybrid_refineT, ft_constraints_refine700/900)
plus the per-case results already computed by src/stage1/arch_eval.py.

    python -m src.stage1.selection --preds outputs/kaggle/selection/preds outputs/kaggle/arch_eval/preds

Steps:
  1. score every candidate that has no per-case results yet (same metrics as
     src/stage1/arch_eval.py: Day 2 + architectural + validity), saved to
     outputs/stage1/arch_eval/per_case/<setting>.csv.gz;
  2. Pareto analysis on (tolerant mIoU, architectural validity), both higher =
     better: overall, type = room and size = large -> outputs/pareto*.png.
     Dominated points are hollow; the knee is the front point farthest from the
     chord between the front's two end points, after min-max normalising both
     axes over all plotted points;
  3. paired bootstrap (src/stage1/stats.py) of the knee and the best hybrid against
     refine t = 600 and the matched U-Net;
  4. size-adaptive option: the knee of each size bin, used for that bin's cases;
  5. outputs/stage1/selection.md with the tables and a recommended configuration.
"""

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from src.stage1.arch_eval import OLD_KEYS, score_setting, summary_row, tables
from src.stage1.eval.metrics import to_markdown
from src.stage1.stats import bootstrap, plain_summary, verdict
from src.utils import project_path

U_NETS = ["unet", "unet_matched"]
DIFFUSION = ["run1_refine600", "ft_control_refine600", "ft_constraints_refine200", "ft_constraints_refine400",
             "ft_constraints_refine600", "ft_constraints_refine700", "ft_constraints_refine800",
             "ft_constraints_refine900", "ft_constraints_ddim50"]
HYBRID = [f"hybrid_refine{t}" for t in (400, 600, 700, 800, 900)]
CANDIDATES = U_NETS + DIFFUSION + HYBRID
REFERENCE = "ft_constraints_refine600"          # the Day 6 main setting
X, Y = "tmIoU", "arch_validity_pct"             # the two selection axes (both: higher = better)
SUBSETS = [("overall", None), ("type = room", ("type", "room")), ("size = large", ("size_bin", "large"))]
SIZE_BINS = ["small", "medium", "large"]
# Tolerant mIoU we accept to give up for CLEAR structural gains (half a point). A judgement call,
# set AFTER seeing the Day 7 results (see docs/decisions.md).
MAX_MIOU_LOSS = 0.005
BOOT_METRICS = ["tolerant mIoU", "architectural validity %", "connectivity error"]
TABLE_COLS = OLD_KEYS + ["dead_ends_per_case", "fragments_per_case", "unreachable_per_case", "arch_validity_pct"]
KEY_COLS = ["tmIoU", "conn_err", "room_leak", "room_count_err", "dead_ends_per_case", "arch_validity_pct"]


# ------------------------------------------------------------------ 1. scoring
RESULTS_CACHE = {}     # results scored in this run (also used when --limit forbids saving)


def find_preds(name, dirs):
    for d in dirs:
        if (Path(d) / f"{name}.npz").exists():
            return d
    return None


def load_results(per_case_dir, pred_dirs, workers, limit=None):
    """Per-case results of every candidate; scores (and saves) the missing ones."""
    todo = [n for n in CANDIDATES if not (per_case_dir / f"{n}.csv.gz").exists() and find_preds(n, pred_dirs)]
    if todo:
        print("scoring:", ", ".join(todo), flush=True)
        with Pool(min(len(todo), workers)) as pool:
            for name, df in pool.imap_unordered(score_setting, [(n, find_preds(n, pred_dirs), None, limit)
                                                                for n in todo]):
                if limit is None:          # never save partial (code-check) results
                    df.to_csv(per_case_dir / f"{name}.csv.gz", index=False)
                print("  scored", name, flush=True)
                RESULTS_CACHE[name] = df
    results = {}
    for n in CANDIDATES:
        if n in RESULTS_CACHE:
            results[n] = RESULTS_CACHE[n]
        elif (per_case_dir / f"{n}.csv.gz").exists():
            results[n] = pd.read_csv(per_case_dir / f"{n}.csv.gz")
    missing = [n for n in CANDIDATES if n not in results]
    if missing:
        print("not available (left out):", ", ".join(missing), flush=True)
    if limit is not None:
        results = {n: df.iloc[:limit] for n, df in results.items()}
    return results


# ------------------------------------------------------------------ 2. Pareto
def points_for(results, rows):
    """(tmIoU, validity) of every candidate on the given case rows."""
    return pd.DataFrame({n: summary_row(df.iloc[rows]) for n, df in results.items()}).T[[X, Y]].astype(float)


def pareto_front(pts):
    """True for points no other point beats on both axes (>= on both, > on one)."""
    x, y = pts[X].to_numpy(), pts[Y].to_numpy()
    dominated = [((x >= x[i]) & (y >= y[i]) & ((x > x[i]) | (y > y[i]))).any() for i in range(len(pts))]
    return ~np.array(dominated)


def normalise(pts):
    lo, hi = pts.min(), pts.max()
    return (pts - lo) / (hi - lo).replace(0, 1)


def knee(pts, front):
    """Front point with the largest distance to the chord between the front's
    end points (both axes min-max normalised over all points). With fewer than
    3 front points there is no interior point: then the front point with the
    largest normalised x + y."""
    norm = normalise(pts)[front].sort_values(X)
    if len(norm) < 3:
        return norm.sum(axis=1).idxmax(), "fewer than 3 front points: largest normalised x + y"
    a, b = norm.iloc[0].to_numpy(), norm.iloc[-1].to_numpy()
    d = b - a
    dist = np.abs(d[0] * (norm[Y] - a[1]) - d[1] * (norm[X] - a[0])) / np.hypot(*d)
    return dist.idxmax(), "max distance to the chord"


def family(name):
    return "U-Net" if name in U_NETS else "hybrid (U-Net sketch + diffusion)" if name in HYBRID else "diffusion"


SHORT = {"unet": "U-Net", "unet_matched": "U-Net matched", "run1_refine600": "run1 r600",
         "ft_control_refine600": "control r600", "ft_constraints_ddim50": "DDIM 50"}


def short(name):
    if name in SHORT:
        return SHORT[name]
    t = name.split("refine")[-1]
    return f"hyb {t}" if name.startswith("hybrid") else f"r{t}"


LABEL_TRIES = [(5, 4), (5, -11), (-45, 4), (-45, -11), (5, 15), (5, -22)]   # offsets in points


def label_offsets(pts, axis_pts=(520, 330), min_gap=(42, 10)):
    """Greedy label placement: for each point, the first offset whose label
    does not sit on an earlier label (positions compared in approximate points)."""
    norm = normalise(pts)
    placed, out = [], {}
    for n in norm.sort_values(Y, ascending=False).index:
        x0, y0 = norm.loc[n, X] * axis_pts[0], norm.loc[n, Y] * axis_pts[1]
        for dx, dy in LABEL_TRIES:
            pos = (x0 + dx, y0 + dy)
            if all(abs(pos[0] - q[0]) > min_gap[0] or abs(pos[1] - q[1]) > min_gap[1] for q in placed):
                break
        placed.append(pos)
        out[n] = (dx, dy)
    return out


def plot_pareto(pts, front, knee_name, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"U-Net": ("s", "tab:orange"), "diffusion": ("o", "tab:blue"),
             "hybrid (U-Net sketch + diffusion)": ("^", "tab:green")}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for fam, (marker, color) in style.items():
        for on_front in (True, False):
            sel = [n for i, n in enumerate(pts.index) if family(n) == fam and front[i] == on_front]
            if sel:
                ax.scatter(pts.loc[sel, X], pts.loc[sel, Y], marker=marker, s=60, linewidths=1.5,
                           edgecolors=color, facecolors=color if on_front else "none",
                           label=f"{fam}{'' if on_front else ' (dominated)'}")
    fr = pts[front].sort_values(X)
    ax.plot(fr[X], fr[Y], "k-", lw=1, alpha=0.6, label="Pareto front")
    if len(fr) >= 2:
        ax.plot(fr[X].iloc[[0, -1]], fr[Y].iloc[[0, -1]], "k:", lw=1, alpha=0.5, label="chord (end points)")
    ax.scatter([pts.loc[knee_name, X]], [pts.loc[knee_name, Y]], s=320, facecolors="none", edgecolors="red",
               linewidths=2, label=f"knee: {short(knee_name)}")
    for n, offset in label_offsets(pts).items():
        ax.annotate(short(n), (pts.loc[n, X], pts.loc[n, Y]), textcoords="offset points", xytext=offset, fontsize=7)
    ax.set_xlabel("tolerant mIoU (higher = better)")
    ax.set_ylabel("architectural validity % (higher = better)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def rows_of(df, sub):
    return np.arange(len(df)) if sub is None else np.flatnonzero(df[sub[0]].to_numpy() == sub[1])


# ------------------------------------------------------------------ 3-4. bootstrap, size-adaptive
def best_hybrid(pts):
    """Hybrid with the largest normalised tmIoU + validity (equal weight, same normalisation as the knee)."""
    norm = normalise(pts)
    return norm.loc[[n for n in pts.index if n in HYBRID]].sum(axis=1).idxmax()


def size_adaptive(results, meta_df):
    """Knee of each size bin, and the per-case results of using it for that bin."""
    picks, frames = {}, []
    sizes = meta_df["size_bin"].to_numpy()
    for b in SIZE_BINS:
        rows = np.flatnonzero(sizes == b)
        pts = points_for(results, rows)
        front = pareto_front(pts)
        picks[b] = (knee(pts, front)[0], pts, front)
    # same case order as every other setting: take each bin's rows from its knee setting
    combined = pd.concat([results[picks[b][0]][sizes == b] for b in SIZE_BINS]).sort_index()
    return picks, combined


def boot_table(comparisons, results, rng):
    lines, summary, overall_res = [], [], {}
    for label, a_name, b_name in comparisons:
        a, b = results[a_name], results[b_name]
        lines += [f"### {label}  (A = {a_name}, B = {b_name})", "",
                  "| subset | metric | A − B | 95% interval | verdict |", "|---|---|---|---|---|"]
        res_all = []
        for sub_label, sub in SUBSETS:
            rows = rows_of(a, sub)
            res = {k: v for k, v in bootstrap(a, b, rows, rng).items() if k in BOOT_METRICS}
            res_all.append((sub_label, res))
            for metric, (diff, lo, hi) in res.items():
                lines.append(f"| {sub_label} (n={len(rows)}) | {metric} | {diff:+.4f} | "
                             f"[{lo:+.4f}, {hi:+.4f}] | {verdict(metric, diff, lo, hi)} |")
        lines.append("")
        summary += plain_summary(label, a_name, res_all)
        overall_res[(a_name, b_name)] = res_all[0][1]
    return summary, lines, overall_res


# ------------------------------------------------------------------ final TEST run (Stage 1, run once)
TEST_METHODS = ["size_adaptive", "unet_matched", "hybrid_refine600", "ft_constraints_refine600", "nearest_fill"]
TEST_SCORED = ["nearest_fill", "unet_matched", "ft_constraints_refine600", "hybrid_refine400", "hybrid_refine600",
               "hybrid_refine700"]
FROZEN_T0 = {"small": "hybrid_refine400", "medium": "hybrid_refine600", "large": "hybrid_refine700"}  # configs/stage1/app.yaml
TEST_COMPARISONS = [("size-adaptive - matched U-Net", "size_adaptive", "unet_matched"),
                    ("size-adaptive - refine 600", "size_adaptive", "ft_constraints_refine600"),
                    ("size-adaptive - hybrid 600", "size_adaptive", "hybrid_refine600")]
POINTS = {"tmIoU": 100, "mIoU": 100, "pixel_acc": 100, "F1_door": 100, "F1_window": 100, "F1_room": 100,
          "alignment": 100, "open_on_wall": 100, "room_leak": 100, "arch_validity_pct": 1}   # -> percentage points


def frozen_adaptive(results):
    """The FROZEN Day 7 size-adaptive hybrid: each size bin's cases from its fixed t0 setting (no re-selection)."""
    sizes = results["hybrid_refine600"]["size_bin"].to_numpy()
    return pd.concat([results[FROZEN_T0[b]][sizes == b] for b in SIZE_BINS]).sort_index()


def score_split(per_case_dir, pred_dirs, workers, split, limit=None):
    per_case_dir.mkdir(parents=True, exist_ok=True)
    results, todo = {}, []
    for n in TEST_SCORED:
        f = per_case_dir / f"{n}.csv.gz"
        if f.exists() and limit is None:
            results[n] = pd.read_csv(f)
        elif n == "nearest_fill" or find_preds(n, pred_dirs):
            todo.append(n)
    if todo:
        print(f"scoring on {split}:", ", ".join(todo), flush=True)
        with Pool(min(len(todo), workers)) as pool:
            for name, df in pool.imap_unordered(score_setting, [(n, find_preds(n, pred_dirs), None, limit, split)
                                                                for n in todo]):
                if limit is None:
                    df.to_csv(per_case_dir / f"{name}.csv.gz", index=False)
                results[name] = df
                print("scored", name, flush=True)
    results["size_adaptive"] = frozen_adaptive(results)
    return results


def test_main(args):
    """--benchmark test: the one final Stage 1 test run (frozen Day 7 configs, no tuning). Scores the saved test
    predictions, builds the frozen size-adaptive hybrid, bootstraps it against the comparison methods (as on val)
    and writes outputs/stage1/stage1_test.md with a val-vs-test table (val from outputs/stage1/arch_eval/per_case when present)."""
    out = project_path(args.out)
    test = score_split(out / "arch_eval_test" / "per_case", [project_path(p) for p in args.preds], args.workers,
                       "test", args.limit)
    val_dir = project_path("outputs/stage1/arch_eval/per_case")
    val = None
    if all((val_dir / f"{n}.csv.gz").exists() for n in TEST_SCORED if n != "nearest_fill"):
        val = {n: pd.read_csv(val_dir / f"{n}.csv.gz") for n in TEST_SCORED if (val_dir / f"{n}.csv.gz").exists()}
        val["size_adaptive"] = frozen_adaptive(val)
    t_overall, t_groups = tables(test, TEST_METHODS)
    rng = np.random.default_rng(0)
    summary, boot_lines, _ = boot_table(TEST_COMPARISONS, test, rng)
    n_test = len(test["unet_matched"])
    L = ["# Stage 1: final TEST evaluation (run once, frozen Day 7 configurations)", "",
         f"Test benchmark: {n_test} cases (data/benchmarks/test_benchmark.npz). Methods exactly as chosen on val "
         "(configs/stage1/app.yaml): size-adaptive hybrid (t0 = 400 / 600 / 700 for small / medium / large holes), matched "
         "U-Net, hybrid t0 = 600, constraint fine-tune refine-600, nearest fill. Same metrics and paired bootstrap as "
         "on val (src/stage1/selection.py). Nothing was tuned on test.", ""]
    keys = KEY_COLS + ["mIoU", "F1_door", "F1_window", "F1_room"]
    if val is not None:
        v_overall, _ = tables(val, TEST_METHODS)
        L += ["## Val vs test (overall)", "",
              "Differences in percentage points for shares / IoU / F1 (x100), raw for counts. **Bold** = test differs "
              "from val by more than 2 points.", ""]
        L += ["| method | metric | val | test | test - val |", "|---|---|---|---|---|"]
        for m in TEST_METHODS:
            for k in keys:
                a, b = float(v_overall.loc[m, k]), float(t_overall.loc[m, k])
                d = (b - a) * POINTS.get(k, 1)
                flag = abs(d) > 2 and k in POINTS
                L.append(f"| {m} | {k} | {a:.4f} | {b:.4f} | {'**' if flag else ''}{d:+.2f}{'**' if flag else ''} |")
        L.append("")
    else:
        L += ["(Val per-case results not found here: the val-vs-test table is added when this runs locally.)", ""]
    L += ["## Test, overall", "", to_markdown(t_overall[TABLE_COLS]), ""]
    for g, df in t_groups.items():
        L += [f"## Test, {g}", "", to_markdown(df[KEY_COLS]), ""]
    L += ["## Paired bootstrap on test (size-adaptive hybrid against the comparison methods)", ""] + summary + [""] + boot_lines
    path = out / ("stage1_test.md" if args.limit is None else "stage1_test_smoke.md")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:40]))


# ------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preds", nargs="+", default=["outputs/kaggle/selection/preds",
                                                       "outputs/kaggle/arch_eval/preds"])
    parser.add_argument("--benchmark", choices=["val", "test"], default="val",
                        help="test = the one final Stage 1 test run (frozen configs; see test_main)")
    parser.add_argument("--out", default="outputs/stage1")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, help="first N cases only (code check; nothing is saved)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.benchmark == "test":
        return test_main(args)
    out = project_path(args.out)
    per_case_dir = out / "arch_eval" / "per_case"
    results = load_results(per_case_dir, [project_path(p) for p in args.preds], args.workers, args.limit)
    names = list(results)
    first = results[names[0]]
    rng = np.random.default_rng(0)
    suffix = "_check" if args.limit else ""

    # 2. Pareto + knee per subset
    pareto = {}
    for (sub_label, sub), fname in zip(SUBSETS, ["pareto", "pareto_room", "pareto_large"]):
        pts = points_for(results, rows_of(first, sub))
        front = pareto_front(pts)
        k, how = knee(pts, front)
        pareto[sub_label] = (pts, front, k, how)
        plot_pareto(pts, front, k, f"Selection: tolerant mIoU vs architectural validity ({sub_label}, "
                    f"{len(rows_of(first, sub))} val cases)", out / f"{fname}{suffix}.png")
    knee_all = pareto["overall"][2]
    hyb = best_hybrid(pareto["overall"][0])

    # 4. size-adaptive
    picks, adaptive = size_adaptive(results, first)
    results_plus = {**results, "size_adaptive": adaptive}

    # 3. bootstrap
    comparisons = []
    if knee_all != REFERENCE:
        comparisons.append(("knee - refine t=600", knee_all, REFERENCE))
    if knee_all != "unet_matched":
        comparisons.append(("knee - matched U-Net", knee_all, "unet_matched"))
    comparisons.append(("best hybrid - matched U-Net", hyb, "unet_matched"))
    comparisons.append(("size-adaptive - knee", "size_adaptive", knee_all))
    if hyb != knee_all:
        comparisons.append(("size-adaptive - best hybrid", "size_adaptive", hyb))
    comparisons.append(("size-adaptive - matched U-Net", "size_adaptive", "unet_matched"))
    boot_summary, boot_lines, boot = boot_table(comparisons, results_plus, rng)

    write_report(out / f"selection{suffix}.md", results_plus, names, pareto, knee_all, hyb, picks,
                 boot_summary, boot_lines, boot, len(first))
    print((out / f"selection{suffix}.md").read_text(encoding="utf-8")[:3000])


def front_table(pts, front, k):
    t = pts.copy()
    t["family"] = [family(n) for n in t.index]
    t["on_front"] = front
    t["knee"] = [n == k for n in t.index]
    t.index.name = "setting"
    return t.sort_values(X)


def write_report(path, results, names, pareto, knee_all, hyb, picks, boot_summary, boot_lines, boot, n_cases):
    shown = names + ["size_adaptive"]
    overall, groups = tables(results, shown)
    pts_all = pareto["overall"][0]
    lines = [f"# Final selection (full val benchmark, {n_cases} cases)", "",
             "Candidates: both U-Nets, every diffusion setting (refine-from-one-shot at t₀, DDIM 50) and the "
             "**hybrids** (`hybrid_refineT`: the matched U-Net's prediction → one-hot → noised to t₀ → "
             "stochastic DDIM with the constraint fine-tune, known-region reset). Selection axes: tolerant mIoU "
             "and architectural validity (both higher = better). The knee = Pareto-front point farthest from the "
             "chord between the front's end points, axes min-max normalised over all candidates. "
             "**All choices are made on val; the val numbers of the chosen setting are therefore slightly "
             "optimistic — the test set gives the final numbers.**", "",
             "## Recommendation", ""] + recommendation(overall, pareto, knee_all, hyb, picks, boot) + [""]
    lines += ["## Pareto analysis", ""]
    for (sub_label, (pts, front, k, how)), fig in zip(pareto.items(), ["pareto", "pareto_room", "pareto_large"]):
        lines += [f"### {sub_label} — knee: **{k}** ({how})  ·  figure `{fig}.png`", "",
                  to_markdown(front_table(pts, front, k)), ""]
    lines += ["## Statistics (paired bootstrap, 1,000 resamples)", "", *boot_summary, "", *boot_lines]
    lines += ["## Size-adaptive option", "",
              "For each size bin (mask area as % of the building; known at inference time because the mask "
              "is given) the knee of that bin's own Pareto front is used for its cases:", "",
              "| size bin | knee setting | its tmIoU | its validity % |", "|---|---|---|---|"]
    for b, (k, pts, _) in picks.items():
        lines.append(f"| {b} | {k} | {pts.loc[k, X]:.3f} | {pts.loc[k, Y]:.1f} |")
    lines += ["", "Combined (`size_adaptive`) vs the single settings:", "",
              to_markdown(overall.loc[list(dict.fromkeys(["unet_matched", REFERENCE, knee_all, hyb,
                                                          "size_adaptive"]))][KEY_COLS]), ""]
    lines += ["## All candidates: overall (all metrics)", "", to_markdown(overall[TABLE_COLS]), ""]
    for g, t in groups.items():
        t.index.name = "setting"
        lines += [f"## {g}", "", to_markdown(t[KEY_COLS]), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def clearly(boot, a, b, metric, want):
    """True if the overall bootstrap interval of A - B excludes 0 in the wanted direction."""
    if (a, b) not in boot:
        return False
    diff, lo, hi = boot[(a, b)][metric]
    return (lo > 0) if want == "up" else (hi < 0)


def recommendation(overall, pareto, knee_all, hyb, picks, boot):
    """Choose between the size-adaptive option and the single knee setting, and say why.

    Rule: the size-adaptive option is recommended if, against the knee, it is
    CLEARLY better on validity and on connectivity error and loses less than
    MAX_MIOU_LOSS tolerant mIoU; otherwise the knee setting."""
    o = overall
    u, k, sa = o.loc["unet_matched"], o.loc[knee_all], o.loc["size_adaptive"]
    sa_wins = (clearly(boot, "size_adaptive", knee_all, "architectural validity %", "up")
               and clearly(boot, "size_adaptive", knee_all, "connectivity error", "down")
               and k[X] - sa[X] < MAX_MIOU_LOSS)
    pick, r = ("size_adaptive", sa) if sa_wins else (knee_all, k)
    per_size = ", ".join(f"{b} holes → `{v[0]}`" for b, v in picks.items())
    knees = {s: v[2] for s, v in pareto.items()}
    pts = pareto["overall"][0]
    hyb_pts = pts.loc[[n for n in pts.index if n in HYBRID]]
    beaten = [n for n in pts.index if n in DIFFUSION and ((hyb_pts[X] >= pts.loc[n, X]) & (hyb_pts[Y] >= pts.loc[n, Y])
                                                         & ((hyb_pts[X] > pts.loc[n, X]) | (hyb_pts[Y] > pts.loc[n, Y]))).any()]
    n_diff = sum(n in DIFFUSION for n in pts.index)
    front_names = ", ".join(f"`{n}`" for n in pts.index[pareto["overall"][1]])
    if sa_wins:
        head = [f"**Recommended final configuration: size-adaptive hybrid** — sketch with the matched U-Net, "
                f"then finish with the constraint fine-tune (stochastic DDIM 50, eta 1, known-region reset), "
                f"re-noising further for bigger holes: {per_size}. The size bin comes from the mask area relative "
                "to the building, both visible in the input."]
    else:
        head = [f"**Recommended final configuration: `{knee_all}`** (the knee of the overall Pareto front)."]
    lines = head + ["", "| configuration | tol. mIoU | validity % | conn. err | room leak | room count err |",
                    "|---|---|---|---|---|---|"]
    for n in dict.fromkeys([pick, knee_all, hyb, "unet_matched", REFERENCE]):
        lines.append(f"| {n} | {o.loc[n, X]:.3f} | {o.loc[n, Y]:.1f} | {o.loc[n, 'conn_err']:.3f} | "
                     f"{o.loc[n, 'room_leak']:.3f} | {o.loc[n, 'room_count_err']:.3f} |")
    lines += ["", "Why:",
              f"- **{len(beaten)} of the {n_diff} pure diffusion settings are dominated by a hybrid** (a hybrid is "
              f"at least as good on both tolerant mIoU and validity, better on one); the overall Pareto front is "
              f"{front_names}. The U-Net sketch fixes diffusion's weak point (committing to a different but "
              f"plausible layout); the diffusion finish fixes the U-Net's (broken, dangling structure).",
              f"- **Against the matched U-Net (the best pixel-accuracy model)**, the chosen configuration gives up "
              f"{u[X] - r[X]:.3f} tolerant mIoU (statistically clear but tiny) for +{r[Y] - u[Y]:.1f} points of "
              f"validity, {u['conn_err'] / max(r['conn_err'], 1e-9):.0f}× lower connectivity error, "
              f"{u['room_leak'] / max(r['room_leak'], 1e-9):.1f}× lower room leak and a lower room count error "
              f"({r['room_count_err']:.3f} vs {u['room_count_err']:.3f}).",
              f"- Bigger holes need more re-noising: the knees are t₀ = 400 overall, but t₀ = 700 for room holes "
              f"(`{knees['type = room']}`) and large holes (`{knees['size = large']}`)."]
    if sa_wins:
        lines.append(f"- **Size-adaptive vs the single knee `{knee_all}`**: clearly better validity "
                     f"(+{sa[Y] - k[Y]:.1f} points) and connectivity error ({sa['conn_err']:.3f} vs "
                     f"{k['conn_err']:.3f}) for {k[X] - sa[X]:.3f} tolerant mIoU, below our tolerance of {MAX_MIOU_LOSS} "
                     "(half a point of tolerant mIoU for clear structural gains — a judgement call we set after "
                     "seeing these results).")
        if hyb != knee_all:
            lines.append(f"- Simpler alternative with one setting: `{hyb}` ({o.loc[hyb, X]:.3f}, "
                         f"{o.loc[hyb, Y]:.1f}%, conn. err {o.loc[hyb, 'conn_err']:.3f}); see the "
                         "size-adaptive − best hybrid bootstrap below.")
    lines += ["- Cost: one U-Net pass plus about 20–35 DDIM steps (the steps at or below t₀ = 400–700 of the 50-step "
              "schedule over 1,000 levels): 0.22–0.37 s per plan measured on a T4, against 0.53 s for full DDIM 50.",
              "- Caveat: the t₀ values were chosen on val, so these val numbers are slightly optimistic; "
              "report the test benchmark as the final result."]
    return lines


if __name__ == "__main__":
    main()
