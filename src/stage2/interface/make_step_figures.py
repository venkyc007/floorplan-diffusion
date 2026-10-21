"""Step-by-step pictures of the final Stage 2 system on gallery drawings (for the README, report, slides).

    python -m src.stage2.interface.make_step_figures     -> outputs/stage2/pipeline_steps/ (+ pipeline_steps.png)

Cases: the FIRST gallery case (src/stage2/interface/make_gallery.py order, held-out val) of each kind in KINDS, no
picking by result. Each runs automatically (PARSE -> CHECK -> FILL -> VERIFY, fast settings); the whole-room case also
runs in manual mode with the gallery's box (what a user would brush). Per case and mode the four step pictures are
saved (parsed map, CHECK flags, completed map, result on the drawing) plus the input with the erased part in orange,
and one row per run goes into the summary figure, titled with the validation's own scores.
"""

import json

import cv2
import numpy as np

from src.stage2.interface import demo
from src.utils import PROJECT_ROOT

KINDS = ["door (a)", "window", "room (whole)", "intact"]
OUT = PROJECT_ROOT / "outputs" / "stage2" / "pipeline_steps"


def save(name, rgb):
    cv2.imwrite(str(OUT / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    gallery = demo.load_gallery()
    runs = []
    for kind in KINDS:
        case = next(c for c in gallery if c["kind"] == kind)
        grey, original, _, _ = demo.load_case(case)
        modes = [("auto", None)] + ([("manual", [tuple(case["box"])])] if kind == "room (whole)" else [])
        for mode, boxes in modes:
            steps = demo.run(grey, boxes=boxes, seed=0)
            s = demo.score(steps["res"], case)
            tag = f"{case['id']:02d}_{kind.replace(' ', '').replace('(', '').replace(')', '')}_{mode}"
            inp = demo.box_on_drawing(grey, case["box"], label=None if case["box"] is None else "erased")
            for name, img in (("input", inp), ("parsed", steps["parsed"]), ("flags", steps["flags"]),
                              ("completed", steps["completed"]), ("result", steps["overlay"])):
                save(f"{tag}_{name}.png", img)
            runs.append({"tag": tag, "kind": kind, "mode": mode, "plan_id": case["plan_id"], "score": s,
                         "messages": steps["messages"], "timings": steps["timings"],
                         "images": [inp, steps["parsed"], steps["flags"], steps["completed"], steps["overlay"]]})
            print(tag, s, steps["messages"][:3], steps["timings"], flush=True)
    titles = ["input (erased part in orange)", "PARSE: 16-class map", "CHECK: flags (red) / your box (blue)",
              "FILL + VERIFY: completed map", "result on the drawing (red)"]
    fig, axes = plt.subplots(len(runs), 5, figsize=(26, 5.2 * len(runs)), squeeze=False)
    for r, run in zip(axes, runs):
        s = run["score"]
        verdict = (f"intact: {s['false_adds']} false additions" if s["intact"] else
                   f"found {'yes' if s['found'] else 'no'}, fixed {'yes' if s['fixed'] else 'no'}, "
                   f"false additions {s['false_adds']}")
        for j, (ax, img) in enumerate(zip(r, run["images"])):
            ax.imshow(img, interpolation="nearest")
            ax.set_title(titles[j] if j else f"{run['kind']} ({run['mode']}) - {verdict}", fontsize=11)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT.parent / "pipeline_steps.png", dpi=60)
    (OUT / "runs.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "images"} for r in runs],
                                              indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                                   encoding="utf-8")
    print("->", OUT)


if __name__ == "__main__":
    main()
