"""The novelty tab of the Gradio app (demo_app/app.py): run one of the hybrid versions on a drawing and show it.

    steps = run(grey, "v4c · final (structurally sound first)")     # automatic: nothing about the omission given

Versions (configs/novelty/app.yaml; each is a frozen pipeline config, all with the same trained models):
  v4c  the final hybrid (round 4): structure net + edit fill + clean-up + repair by the three structural checks,
       best of 4 diffusion samples by soundness
  v4b  v4c with the clean-up's wall-end rule: the most sound on tuning, a little less accurate
  v3c  the round-2 final: accuracy first, no clean-up / repair
Pictures reuse the Stage 2 tab's drawing helpers (src/stage2/interface/demo.py). The structural problems of the
completed plan (src/novelty/soundness.problem_map) are drawn on it, inside the areas the system flagged as regions,
so a viewer can see why a fill is or is not "sound" without the true layout. For gallery cases the run is also scored
with the evaluation's own code (src/novelty/eval_e2e.row_of): found, fixed (strict), recovered (2 px), sound.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.novelty import pipeline as nov
from src.novelty.soundness import problem_map, soundness
from src.stage2.interface import demo as S2
from src.utils import PROJECT_ROOT

CFG = yaml.safe_load((PROJECT_ROOT / "configs" / "novelty" / "app.yaml").read_text(encoding="utf-8"))
VERSIONS = CFG["versions"]
GALLERY_DIR = PROJECT_ROOT / CFG["gallery_dir"]
# drawn as rings (not fills) in colours the 16-class palette does not use, so they never look like a door or fixture
PROBLEM_COLOURS = {"dead_ends": (0, 220, 0), "floating": (255, 215, 0), "bad_openings": (130, 0, 255),
                   "leaks": (0, 230, 255), "doorless": (110, 110, 0), "mixed": (255, 105, 180)}
PROBLEM_NAMES = {"dead_ends": "wall ends in the air", "floating": "wall piece touching nothing",
                 "bad_openings": "door / window not in a wall", "leaks": "room open to the outside",
                 "doorless": "closed room with no door", "mixed": "two room types touching without a wall"}


def checkpoints(path):
    return yaml.safe_load((PROJECT_ROOT / path).read_text(encoding="utf-8"))["checkpoints"]


def use_version(label):
    """Point the hybrid pipeline at this version's frozen config. The trained models are loaded once and shared:
    all versions name the same checkpoints (checked by tests/novelty/test_app_demo.py)."""
    if label not in VERSIONS:
        raise ValueError(f"unknown version {label!r}")
    nov.CFG_FILE = PROJECT_ROOT / VERSIONS[label]


# ---------------------------------------------------------------- gallery

def load_gallery():
    index = GALLERY_DIR / "index.json"
    if not index.exists():
        return []
    cases = json.loads(index.read_text(encoding="utf-8"))
    for c in cases:
        c["drawing"] = str(GALLERY_DIR / f"{c['id']:02d}_drawing.png")
    return cases


def case_caption(c):
    if c["size"] == "intact":
        return "intact (nothing erased)"
    return f"{c['kind']} erased · {c['size']} ({float(c['meta']['size_pct']):.0f}% of the building)"


def load_case(c):
    """(erased, original, true labels, removed mask, erased area) of a gallery case."""
    read = lambda s, f=cv2.IMREAD_GRAYSCALE: cv2.imread(str(GALLERY_DIR / f"{c['id']:02d}_{s}.png"), f)
    return read("drawing"), read("original"), read("labels", cv2.IMREAD_UNCHANGED), read("removed") > 0, read("area") > 0


def heldout_markdown(c):
    """What each version did on this very case in the held-out evaluation (kernel v16)."""
    if not c.get("heldout"):
        return ""
    yes = lambda v: {"True": "yes", "False": "no", "": "–"}.get(v, v)
    if c["size"] == "intact":
        rows = [f"| {v.split(' ·')[0]} | {r['n_region_flags']} | {r['added']} |" for v, r in c["heldout"].items()]
        return ("**This case in the held-out evaluation** (run once per version):\n\n| version | region flags | "
                "pieces added |\n|---|---|---|\n" + "\n".join(rows))
    rows = [f"| {v.split(' ·')[0]} | {yes(r['found'])} | {yes(r['recovered'])} | {yes(r['fixed'])} | "
            f"{yes(r['s_sound'])} |" for v, r in c["heldout"].items()]
    return ("**This case in the held-out evaluation** (run once per version; a live run here may differ a little: "
            "fast settings, other seed):\n\n| version | found | recovered (2 px) | fixed (strict) | sound |\n"
            "|---|---|---|---|---|\n" + "\n".join(rows))


# ---------------------------------------------------------------- run + pictures

def region_area(res):
    """The areas the system flagged as regions (large omissions), in map pixels."""
    m = np.zeros(res["parsed"].shape, bool)
    for h in res["holes"]:
        if h["type"] == "region":
            y0, x0, y1, x1 = h["box"]
            m[y0:y1 + 1, x0:x1 + 1] = True
    return m


def problems_image(plan, region):
    """The completed plan with each structural problem inside the flagged regions drawn in its colour."""
    rgb, k = S2.label_image(plan)
    if not region.any():
        return rgb, {}
    pm = problem_map(plan, region)
    for name, m in pm.items():
        if m.any():
            big = np.repeat(np.repeat(m, k, 0), k, 1).astype(np.uint8)
            big = cv2.dilate(big, np.ones((3, 3), np.uint8), iterations=max(2, k * 2))
            contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, contours, -1, PROBLEM_COLOURS[name], max(2, k + 1))
    ys, xs = np.nonzero(region)
    cv2.rectangle(rgb, (int(xs.min() * k), int(ys.min() * k)), (int((xs.max() + 1) * k) - 1,
                  int((ys.max() + 1) * k) - 1), (0, 90, 255), max(1, k))
    return rgb, {n: int(m.any()) for n, m in pm.items()}


def problems_legend_html():
    chips = [f"<span style='display:inline-block;margin:2px 10px 2px 0'><span style='display:inline-block;width:10px;"
             f"height:10px;border:3px solid rgb{c};border-radius:50%;margin-right:4px;vertical-align:middle'></span>"
             f"{PROBLEM_NAMES[n]}</span>"
             for n, c in PROBLEM_COLOURS.items()]
    return ("<div><b>Structural problems</b> (inside the flagged regions, blue box): " + "".join(chips) + "</div>")


def run(grey, label, seed=0):
    """Run one hybrid version automatically -> pictures + text (as src/stage2/interface/demo.run)."""
    use_version(label)
    t = time.time()
    res = nov.run(grey, seed=int(seed))
    seconds = time.time() - t
    region = region_area(res)
    probs, _ = problems_image(res["completed"], region)
    sound = soundness(res["completed"], region) if region.any() else None
    return {"res": res, "seconds": round(seconds, 1), "sound": sound,
            "parsed": S2.label_image(res["parsed"])[0],
            "flags": S2.flags_image(res["parsed"], res["holes"]),
            "completed": S2.completed_image(res["parsed"], res["completed"]),
            "problems": probs,
            "overlay": S2.fit(res["overlay"]),
            "messages": res["messages"],
            "rows": S2.hole_rows(res["holes"])}


def score(res, case):
    """Score a run on a gallery case with the evaluation's own code (eval_e2e.row_of) -> row dict."""
    from src.novelty.eval_e2e import row_of
    erased, _, lbl, removed, area = load_case(case)
    if case["size"] == "intact":
        return row_of(case["plan_id"], "intact", None, "app", res, 0)
    o = {0: erased, 1: removed, 2: case["expected_type"], 4: {**case["meta"], "type": case["kind"]}, 5: area,
         "lbl": lbl}
    return row_of(case["plan_id"], case["size"], o, "app", res, 0)


def report_markdown(steps, label, case=None):
    res = steps["res"]
    lines = [f"**{label}** · {steps['seconds']} s · {len(res['holes'])} flag(s), "
             f"{sum(h['type'] == 'region' for h in res['holes'])} of them region flags (large omissions)"]
    s = steps["sound"]
    if s is None:
        lines.append("No region flag: the large-omission path did not run (small flags, if any, use the shipped fill).")
    else:
        ok = lambda b: "✅" if b else "❌"
        lines.append(f"**Structural checks of the filled regions** (no ground truth needed): walls connect "
                     f"{ok(s['walls_connect'])} · openings in walls {ok(s['openings_in_walls'])} · rooms closed and "
                     f"reachable {ok(s['rooms_closed'])} → **{'sound' if s['sound'] else 'not sound'}**")
    if case is not None:
        r = score(res, case)
        yes = lambda v: "yes" if v is True or str(v) == "True" else ("–" if v in ("", None) else "no")
        if case["size"] == "intact":
            lines.append(f"**Scored against the truth** (intact drawing): {r['n_region_flags']} region flag(s), "
                         f"{r['added']} piece(s) added (both should be 0).")
        else:
            lines.append(f"**Scored against the truth** (the evaluation's definitions): found **{yes(r['found'])}**, "
                         f"recovered (2 px) **{yes(r['recovered'])}**, fixed (strict, IoU ≥ 0.3) "
                         f"**{yes(r['fixed'])}** (IoU {float(r['fix_iou']):.2f}), sound **{yes(r.get('s_sound', ''))}**"
                         f" (the true layout itself: {yes(r.get('st_sound', ''))}).")
    lines.append("**What the system reports:**")
    lines += [f"- {m}" for m in steps["messages"]] or ["- nothing flagged: the system thinks the drawing is complete"]
    return "\n".join(lines)


def compare_image(results, side=None):
    """Side by side: each version's completed plan with its structural problems (labels on top)."""
    side = side or S2.CFG["display_max_side"]
    tiles = []
    for label, steps in results:
        img = S2.fit(steps["problems"], side // max(1, len(results)) + 80)
        bar = np.full((34, img.shape[1], 3), 255, np.uint8)
        s = steps["sound"]
        text = label.split(" ·")[0] + ("" if s is None else ("  sound" if s["sound"] else "  not sound"))
        cv2.putText(bar, text, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        tiles.append(np.vstack([bar, img]))
    h = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.full((h - t.shape[0], t.shape[1], 3), 255, np.uint8)]) for t in tiles]
    gap = np.full((h, 12, 3), 255, np.uint8)
    out = tiles[0]
    for t in tiles[1:]:
        out = np.hstack([out, gap, t])
    return out


SAVED_FIGURES = PROJECT_ROOT / "outputs" / "novelty" / "assessment"


def saved_figures():
    """[(png path, caption)] of the assessment figures (held-out results of the final hybrid, constraint ablation)."""
    if not SAVED_FIGURES.exists():
        return []
    out = []
    for p in sorted(SAVED_FIGURES.glob("*.png")):
        name = p.stem
        cap = ("held-out large omission (v4c)" if name.startswith("large_heldout") else
               "constraint ablation: the same case through each version" if name.startswith("constraints") else
               "plausible alternatives: sound but different from the truth" if name.startswith("alternatives") else
               name.replace("_", " "))
        out.append((str(p), cap))
    return out
