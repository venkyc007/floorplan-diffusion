"""Figures for the novelty poster and the 2-page report (pictures only; the text is the student's own).

    python -m src.novelty.poster_figures                -> outputs/novelty/poster/

  hybrid_pipeline.png   the full combined architecture of the final hybrid (v4c): what each part is
  novelty_workflow.png  the four rounds of the novelty work, each with its held-out result on large omissions
  results_<split>.png   bars: shipped / v3c / v4b / v4c on large and medium omissions (recovered, fixed, sound),
                        with the true layout's own soundness as a reference line; split = heldout and, once the
                        test comparison exists (kernel v18), test
No drawing of CubiCasa5K is in these figures (diagrams and numbers only), so they can be committed.
Numbers are read from the saved per-case results (outputs/novelty/*.csv), never typed in.
"""

import csv
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from src.utils import PROJECT_ROOT  # noqa: E402

OUT = PROJECT_ROOT / "outputs" / "novelty" / "poster"
NOV = PROJECT_ROOT / "outputs" / "novelty"
# fixed categorical order (validated palette, slots 1-3); the shipped baseline is neutral grey
COLOURS = {"v4c (final)": "#2a78d6", "v3c (accuracy first)": "#eb6834", "v4b (most sound)": "#1baf7a",
           "shipped Stage 2": "#9a9a96"}
INK, MUTED = "#222222", "#666666"
FILL = {"in": "#dbe8f7", "old": "#f3f2ee", "new": "#fde3d4", "check": "#dff2e8", "out": "#e8e4f3"}


def box(ax, x, y, w, h, text, kind, size=10.5, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=FILL[kind],
                                ec="#333333", lw=1.3))
    lines = text.split("\n")
    ax.text(x + w / 2, y + h - 0.22, lines[0], ha="center", va="top", fontsize=size + 1,
            fontweight="bold" if bold_first else "normal", color=INK)
    if len(lines) > 1:
        ax.text(x + w / 2, y + h - 0.62, "\n".join(lines[1:]), ha="center", va="top", fontsize=size, color=INK,
                linespacing=1.35)


def arrow(ax, a, b, dashed=False, colour="#333333"):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=1.5, color=colour,
                                                    linestyle="--" if dashed else "-", mutation_scale=16))


def hybrid_pipeline(path):
    fig, ax = plt.subplots(figsize=(19, 8.2))
    ax.set_xlim(0, 19)
    ax.set_ylim(0.6, 8.6)
    ax.axis("off")
    box(ax, 0.2, 5.3, 2.3, 2.0, "Input drawing\ngreyscale, any size\nnothing about the\nomission is given", "in")
    box(ax, 3.0, 5.3, 2.9, 2.0, "PARSE (Stage 2)\nU-Net, ResNet-34 enc.\ndrawing -> 16-class map", "old")
    box(ax, 6.4, 6.0, 3.6, 2.2, "CHECK (Stage 2)\nrules + detectors v2 / v3\n-> small flags\n(doors, windows, fixtures)",
        "old")
    box(ax, 10.5, 6.0, 3.6, 2.2, "Small-hole FILL (Stage 2)\n16-class U-Net sketch\n-> diffusion finish\n"
        "best of N by rule count", "old")
    box(ax, 6.4, 2.9, 3.6, 2.8, "STRUCTURE NET (new)\nU-Net, ResNet-34 encoder,\nwhole plan, no box\nin: parsed map + wall rays\n"
        "+ ink map + stroke map\nout: missing-wall heat", "new")
    box(ax, 6.4, 0.8, 3.6, 1.8, "REGION FLAGS (new)\nheat >= 0.5 -> centre lines\ngroups of added walls\n= large omissions",
        "new")
    box(ax, 10.5, 0.8, 4.0, 4.9, "Large-hole FILL (new)\nedit-mode U-Net: sees the\nparsed map in the box ->\n"
        "sketch (wall if p >= 0.18)\n+ structure walls painted in\n-> noised to t0 = 700\n-> 16-class diffusion finish\n"
        "(15 DDIM steps)\n4 samples", "new")
    box(ax, 15.0, 3.1, 3.8, 2.6, "FINISH each sample (v4c)\nclean-up: snap / join walls,\ndrop specks and loose bits\n"
        "repair: join floating walls,\nclose rooms, extend ends,\none door per closed room", "check", size=10)
    box(ax, 15.0, 0.8, 3.8, 1.8, "SELECT by soundness\nkeep the sample with the\nfewest structural problems", "check",
        size=10)
    box(ax, 15.0, 6.0, 3.8, 2.2, "Output\ncompleted 16-class plan\nadditions drawn in red\non the drawing + messages",
        "out")
    arrow(ax, (2.5, 6.3), (3.0, 6.3))
    arrow(ax, (5.9, 6.6), (6.4, 7.0))
    arrow(ax, (5.9, 5.8), (6.4, 4.8))
    arrow(ax, (10.0, 7.1), (10.5, 7.1))
    arrow(ax, (14.1, 7.1), (15.0, 7.1))
    arrow(ax, (8.2, 2.9), (8.2, 2.6))
    arrow(ax, (10.0, 1.7), (10.5, 1.7))
    arrow(ax, (10.0, 4.0), (10.5, 4.0), dashed=True)
    arrow(ax, (14.5, 4.0), (15.0, 4.0))
    arrow(ax, (16.9, 3.1), (16.9, 2.6))
    ax.annotate("", xy=(18.9, 6.0), xytext=(18.9, 1.7),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#333333", connectionstyle="arc3,rad=0",
                                mutation_scale=16))
    ax.text(10.25, 4.55, "structure\nwalls", fontsize=9, color=MUTED, ha="center", rotation=0)
    ax.text(8.2, 8.45, "a region flag swallows the small flags inside it", fontsize=9.5, color=MUTED, ha="center")
    for i, (k, t) in enumerate((("old", "Stage 2 (shipped, frozen)"), ("new", "new: trained in the novelty work"),
                                ("check", "new: rule-based (three structural checks)"))):
        ax.add_patch(FancyBboxPatch((0.25, 3.9 - i * 0.6), 0.35, 0.35, boxstyle="round,pad=0.01", fc=FILL[k],
                                    ec="#333333", lw=1))
        ax.text(0.75, 4.07 - i * 0.6, t, va="center", fontsize=10, color=INK)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rows(pattern):
    out = []
    for p in sorted(glob.glob(str(pattern))):
        out += list(csv.DictReader(open(p, encoding="utf-8")))
    return out


def rate(R, key):
    S = [r for r in R if r.get(key, "") != ""]
    return (100 * sum(str(r[key]) == "True" for r in S) / len(S), len(S)) if S else (None, 0)


def arms_for(split):
    """{label: rows} per version for a split, from the saved per-case results; missing versions are left out."""
    if split == "heldout":
        r4 = rows(NOV / "e2e_heldout_round4.csv")
        old = rows(NOV / "e2e_heldout_all_arms.csv")
        return {"shipped Stage 2": [r for r in old if r["arm"] == "shipped auto"],
                "v3c (accuracy first)": [r for r in r4 if r["arm"] == "hybrid v3c auto"],
                "v4b (most sound)": [r for r in r4 if r["arm"] == "hybrid v4b auto"],
                "v4c (final)": [r for r in r4 if r["arm"] == "hybrid v4c auto"]}
    v18 = rows(PROJECT_ROOT / "outputs" / "kaggle" / "novelty_eval18" / "novelty" / "e2e_test_v3c_[01].csv")
    v18b = rows(PROJECT_ROOT / "outputs" / "kaggle" / "novelty_eval18" / "novelty" / "e2e_test_v4b_[01].csv")
    out = {"shipped Stage 2": [r for r in v18 if r["arm"] == "shipped auto"],
           "v3c (accuracy first)": [r for r in v18 if r["arm"] == "hybrid auto"],
           "v4b (most sound)": [r for r in v18b if r["arm"] == "hybrid auto"],
           "v4c (final)": rows(NOV / "e2e_test_round4.csv")}
    return {k: v for k, v in out.items() if v}


def results(split, path):
    arms = arms_for(split)
    metrics = [("recovered", "recovered (2 px)"), ("fixed", "fixed (strict IoU)"), ("s_sound", "sound (3 checks)")]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), sharey=True)
    for ax, size in zip(axes, ("large", "medium")):
        labels = list(arms)
        width = 0.8 / len(labels)
        truth = None
        for j, lab in enumerate(labels):
            R = [r for r in arms[lab] if r["target"] == size]
            for i, (k, _) in enumerate(metrics):
                v, n = rate(R, k)
                if v is None:
                    continue
                x = i + (j - (len(labels) - 1) / 2) * width
                ax.bar(x, v, width * 0.9, color=COLOURS[lab], label=lab if i == 0 else None)
                ax.text(x, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=9.5, color=INK)
            if lab == "v4c (final)":
                truth = rate(R, "st_sound")
        if truth and truth[0] is not None:
            ax.plot([1.6, 2.4], [truth[0]] * 2, color=INK, lw=1.5, ls="--")
            ax.text(1.57, truth[0], f"true layout\nitself: {truth[0]:.0f}%", fontsize=9, color=INK, ha="right",
                    va="center")
        n = len([r for r in arms.get("v4c (final)", []) if r["target"] == size])
        ax.set_title(f"{size} omissions (n = {n} drawings)", fontsize=13, color=INK)
        ax.set_xticks(range(len(metrics)), [t for _, t in metrics], fontsize=11)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", color="#e6e6e6", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("% of cases", fontsize=11)
    axes[0].legend(frameon=False, fontsize=10.5, loc="upper left", ncol=2)
    name = {"heldout": "held-out validation drawings (151)", "test": "official TEST drawings (one run each)"}[split]
    fig.suptitle(f"End to end, nothing given: {name}", fontsize=14, color=INK)
    note = ("recovered: found + walls back within 2 px · fixed: found + IoU >= 0.3 · sound: walls connect, openings in "
            "walls, rooms closed (no ground truth)")
    if split == "heldout":
        note += " · shipped soundness not measured on held-out"
    fig.text(0.5, -0.01, note, ha="center", va="top", fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {lab: {size: {k: rate([r for r in R if r["target"] == size], k)[0] for k, _ in metrics}
                  for size in ("large", "medium")} for lab, R in arms.items()}


def workflow(path):
    """The four rounds, each with its held-out numbers on large omissions (from the saved csv files)."""
    old = rows(NOV / "e2e_heldout_all_arms.csv")
    r4 = rows(NOV / "e2e_heldout_round4.csv")
    L = lambda R: [r for r in R if r["target"] == "large"]
    nums = {"shipped": L([r for r in old if r["arm"] == "shipped auto"]),
            "v2": L([r for r in old if r["arm"] == "hybrid v2 auto"]),
            "v3c": L([r for r in r4 if r["arm"] == "hybrid v3c auto"]),
            "v4c": L([r for r in r4 if r["arm"] == "hybrid v4c auto"])}
    f = lambda k, m: "–" if rate(nums[k], m)[0] is None else f"{rate(nums[k], m)[0]:.0f}%"
    steps = [("Start: shipped Stage 2", "finds 80% of large\nomissions, but its fill\nredraws almost none",
              "shipped", "old"),
             ("Round 1: locate + fill", "whole-plan LOCATOR\n+ large-hole fill retrained\non a large-omission set\n"
              "(edit mode: sees the map)", "v2", "new"),
             ("Round 2: structure first", "STRUCTURE NET marks\nmissing walls (ink + stroke\nmaps, wall rays);\n"
              "walls painted into the fill", "v3c", "new"),
             ("Rounds 3-4: validity", "clean-up + repair by the\nthree structural checks;\n4 samples, most sound kept\n"
              "(review feedback)", "v4c", "check")]
    fig, ax = plt.subplots(figsize=(19, 4.4))
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    for i, (title, body, key, kind) in enumerate(steps):
        x = 0.2 + i * 4.75
        stats = f"recovered {f(key, 'recovered')} · fixed {f(key, 'fixed')}"
        if key in ("v3c", "v4c"):
            stats += f"\nsound {f(key, 's_sound')}"
        box(ax, x, 0.1, 4.1, 3.9, f"{title}\n{body}\n\nheld-out, large:\n{stats}", kind, size=11)
        if i < 3:
            arrow(ax, (x + 4.1, 2.05), (x + 4.75, 2.05))
    ax.text(9.5, 4.35, "Every setting chosen on 212 tuning drawings; each version frozen, then run once on 151 "
            "held-out drawings", ha="center", va="top", fontsize=10, color=MUTED)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hybrid_pipeline(OUT / "hybrid_pipeline.png")
    workflow(OUT / "novelty_workflow.png")
    for split in ("heldout", "test"):
        if split == "test" and len(arms_for("test")) < 4:
            print("test comparison not complete yet (kernel v18): results_test.png skipped")
            continue
        print(split, results(split, OUT / f"results_{split}.png"))
    print("->", OUT)


if __name__ == "__main__":
    main()
