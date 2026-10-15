"""Gradio interface: repair missing parts of a floor plan.

    python app/app.py            (then open http://127.0.0.1:7860)

Steps on the page:
  1. choose a plan: a val-benchmark sample (true plan known), the plan editor
     (draw/delete walls, doors, windows, rooms, outdoor, background), or an
     upload (PNG in our class colours, or a .npy label map);
  2. brush the region to repair (grey); optionally let the app suggest
     suspicious areas from our architectural rules (assistive only);
  3. pick a method and a seed, and repair.
All model work happens in src/interface/ (this file only builds the page).
Settings: configs/app.yaml (checkpoint paths, frozen sampling settings).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                    # so `src` imports work with "python app/app.py"

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.data.classes import BACKGROUND, CLASSES, COLORS, WALL  # noqa: E402
from src.data.corruptions import UNKNOWN  # noqa: E402
from src.interface.inference import (METHOD_NAMES, STOCHASTIC, CheckpointMissing, Repairer,  # noqa: E402
                                     describe, load_config)
from src.interface.plan_io import InputError, load_file  # noqa: E402
from src.interface.render import (difference_image, editor_to_labels, layers_to_mask,  # noqa: E402
                                  mask_layer, plan_image)
from src.interface.report import gt_scores, rule_table, to_markdown  # noqa: E402
from src.interface.suggest import suggest  # noqa: E402
from src.utils import project_path  # noqa: E402

CFG = load_config()
SCALE = CFG["display_scale"]
SIDE = 128 * SCALE
THRESHOLDS = yaml.safe_load(open(project_path("configs/arch_metrics.yaml"), encoding="utf-8"))
LABEL_TO_METHOD = {METHOD_NAMES[m]: m for m in CFG["methods"]}
# Plans are tiny pixel maps: show them with sharp pixels instead of browser smoothing.
CSS = "img, canvas { image-rendering: pixelated; }"


def hex_color(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# ------------------------------------------------------------------ start-up
def load_models():
    """Check that both checkpoints exist (they are loaded later, on first use);
    stop with a clear message if one is missing."""
    try:
        return Repairer(CFG)
    except CheckpointMissing as e:
        print(f"\nERROR: {e}\n", file=sys.stderr)
        sys.exit(1)


def load_gallery():
    path = project_path(CFG["gallery"])
    if not path.exists():
        print(f"Note: no sample gallery at {path} - make it with: python -m src.interface.make_gallery")
        return None
    g = np.load(path)
    return {k: g[k] for k in g.files}


# ------------------------------------------------------------------ small helpers
def new_state(plan, gt=None, source="", preset_mask=None):
    return {"plan": plan, "gt": gt, "source": source, "preset_mask": preset_mask}


def mask_editor_value(plan, mask=None):
    layer = mask_layer(mask if mask is not None else np.zeros(plan.shape, bool), SCALE)
    return {"background": plan_image(plan, SCALE), "layers": [layer], "composite": None}


def plan_status(state, note=""):
    plan = state["plan"]
    unknown = int((plan == UNKNOWN).sum())
    text = f"✅ **Current plan:** {state['source']}"
    if state["gt"] is not None:
        text += " · true plan known (accuracy will be shown)"
    if unknown:
        text += f" · {unknown} grey (unknown) pixels — always repaired"
    return text + (f"\n\n{note}" if note else "")


def current_mask(state, mask_value):
    """The brushed mask (128x128 bool); unknown (grey) pixels are always included."""
    layers = (mask_value or {}).get("layers") if isinstance(mask_value, dict) else None
    mask = layers_to_mask(layers, SCALE, CFG["mask_coverage"])
    if mask is None or mask.shape != state["plan"].shape:
        mask = np.zeros(state["plan"].shape, bool)
    return mask | (state["plan"] == UNKNOWN)


# ------------------------------------------------------------------ step 1: input
def pick_sample(evt: gr.SelectData):
    i = evt.index
    plan, mask, gt = GALLERY["inputs"][i], GALLERY["masks"][i], GALLERY["targets"][i]
    state = new_state(plan.copy(), gt.copy(), f"val sample {i + 1} (plan {GALLERY['plan_ids'][i]}, "
                      f"{GALLERY['types'][i]} removed, {GALLERY['size_bins'][i]} hole)", mask.copy())
    return state, mask_editor_value(plan, mask), plan_status(state, "Its benchmark mask is pre-filled in step 2.")


def editor_start(state, blank):
    if blank:
        plan = np.full((128, 128), BACKGROUND, np.uint8)
    elif state is None:
        raise gr.Error("No plan loaded yet: pick a sample first, or start blank.")
    else:
        plan = state["plan"]
    return {"background": plan_image(plan, SCALE), "layers": [], "composite": None}


def editor_use(value):
    if not isinstance(value, dict) or value.get("composite") is None:
        raise gr.Error("The editor is empty: start from the current plan or a blank canvas first.")
    plan = editor_to_labels(np.asarray(value["composite"]), SCALE, CFG["upload"]["unknown_distance"])
    walls = int((plan == WALL).sum())
    if walls < CFG["upload"]["min_wall_px"]:
        raise gr.Error(f"Only {walls} wall pixels: draw a building with walls (at least "
                       f"{CFG['upload']['min_wall_px']}) before using the plan.")
    state = new_state(plan, None, "edited plan (no true plan, so no accuracy scores)")
    return state, mask_editor_value(plan), plan_status(state)


def upload_use(file, state):
    if file is None:
        raise gr.Error("Choose a .png or .npy file first.")
    try:
        plan = load_file(file, CFG["upload"])
    except InputError as e:
        gr.Warning(f"Upload not usable: {e}")
        keep = plan_status(state) if state is not None else ""
        return state, gr.update(), f"❌ **Upload not usable:** {e}\n\n{keep}"
    new = new_state(plan, None, f"uploaded file {Path(file).name} (framed on the building, 128×128)")
    return new, mask_editor_value(plan), plan_status(new)


# ------------------------------------------------------------------ step 2: mask
def suggest_areas(state):
    if state is None:
        raise gr.Error("Choose a plan first (step 1).")
    mask, found = suggest(state["plan"], THRESHOLDS, CFG["suggest_padding_px"])
    if not found:
        note = ("No rule violations found (this does **not** mean the plan is complete: the check covers only "
                "4 rules). Brush the region yourself.")
    else:
        counts = {}
        for name, _ in found:
            counts[name] = counts.get(name, 0) + 1
        note = ("**Suggested** areas (assistive, not exhaustive — review and edit before repairing): "
                + ", ".join(f"{n} × {k}" for k, n in counts.items()) + ".")
    return mask_editor_value(state["plan"], mask), note


def clear_mask(state):
    if state is None:
        raise gr.Error("Choose a plan first (step 1).")
    return mask_editor_value(state["plan"])


# ------------------------------------------------------------------ step 3: repair
def repair(state, mask_value, method_label, seed, progress=gr.Progress()):
    if state is None:
        raise gr.Error("Choose a plan first (step 1).")
    mask = current_mask(state, mask_value)
    if not mask.any():
        raise gr.Error("Brush over the region to repair first (step 2), or use 'Suggest suspicious areas'.")
    method = LABEL_TO_METHOD[method_label]
    plan = state["plan"]
    result = REPAIRER.repair(plan, mask, method, int(seed),
                             progress=lambda done, total, text: progress(done / max(total, 1), desc=text))
    gt = state["gt"]
    scores = gt_scores(result.labels, gt, mask) if gt is not None else None
    report = to_markdown(rule_table(result.labels, mask, THRESHOLDS, gt), scores, result.seconds, describe(result))
    return (plan_image(plan, SCALE, mask), plan_image(result.labels, SCALE),
            difference_image(plan, result.labels, mask, SCALE), report)


def repair_another(state, mask_value, method_label, seed, progress=gr.Progress()):
    if LABEL_TO_METHOD[method_label] not in STOCHASTIC:
        gr.Info("This method is deterministic: a new seed gives the same result.")
    seed = int(seed) + 1
    return (seed, *repair(state, mask_value, method_label, seed, progress))


# ------------------------------------------------------------------ page
def legend_html():
    names = {"wall": "wall", "door": "door", "window": "window", "room": "room", "outdoor": "outdoor",
             "background": "background"}
    chips = "".join(f'<span style="display:inline-block;margin-right:12px"><span style="display:inline-block;'
                    f'width:14px;height:14px;background:{hex_color(COLORS[c])};border:1px solid #999;'
                    f'vertical-align:middle"></span> {names[CLASSES[c]]}</span>' for c in range(len(CLASSES)))
    grey = ('<span style="display:inline-block"><span style="display:inline-block;width:14px;height:14px;'
            'background:#969696;border:1px solid #999;vertical-align:middle"></span> region to repair / unknown'
            '</span>')
    return f"<div>{chips}{grey}</div>"


def build_ui():
    brush_colors = [hex_color(COLORS[c]) for c in range(len(CLASSES))]
    with gr.Blocks(title="Floor plan repair") as demo:
        state = gr.State(None)
        gr.Markdown("# Floor plan repair\nReconstruct missing walls, doors, windows and rooms. "
                    "**1** choose a plan → **2** brush the region to repair → **3** repair. "
                    f"Running on **{REPAIRER.device.type.upper()}**.")
        gr.HTML(legend_html())
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 1 · Plan")
                with gr.Tabs():
                    with gr.Tab("Samples (val set)"):
                        if GALLERY is None:
                            gr.Markdown("No sample gallery found. Make it with "
                                        "`python -m src.interface.make_gallery`.")
                            gallery = None
                        else:
                            items = [(plan_image(p, 2, m), f"{t} · {s}") for p, m, t, s in
                                     zip(GALLERY["inputs"], GALLERY["masks"], GALLERY["types"], GALLERY["size_bins"])]
                            gallery = gr.Gallery(items, columns=4, height=520, allow_preview=False,
                                                 label="24 val-benchmark cases (caption: removed part · hole size)")
                    with gr.Tab("Editor"):
                        gr.Markdown("Draw with the class colours; to **delete** something, paint over it with "
                                    "room, outdoor or background colour. Then press *Use edited plan*.")
                        with gr.Row():
                            start_current = gr.Button("Start from current plan", size="sm")
                            start_blank = gr.Button("Start blank", size="sm")
                        editor = gr.ImageEditor(
                            type="numpy", image_mode="RGB", sources=(), transforms=(), layers=False,
                            canvas_size=(SIDE, SIDE), fixed_canvas=True, height=SIDE + 120, width=SIDE + 40,
                            brush=gr.Brush(default_size=8, colors=brush_colors, default_color=brush_colors[WALL],
                                           color_mode="fixed"),
                            label="Plan editor (128 × 128 plan shown at 4×)")
                        editor_btn = gr.Button("Use edited plan", variant="primary")
                    with gr.Tab("Upload"):
                        gr.Markdown("A **PNG** in our 6 class colours (each pixel snaps to the nearest class colour; "
                                    "grey = unknown) or a **.npy** label map (values 0–5 = "
                                    + ", ".join(CLASSES) + "; 255 = unknown). The plan is framed on its walls, "
                                    "doors and windows and resized to 128 × 128 (nearest neighbour), as in our data.")
                        upload = gr.File(file_types=[".png", ".npy", ".jpg", ".jpeg", ".bmp", ".webp"],
                                         type="filepath", label="Plan file")
                        upload_btn = gr.Button("Use uploaded file", variant="primary")
                status = gr.Markdown("Choose a plan to start.")
            with gr.Column():
                gr.Markdown("## 2 · Region to repair")
                mask_editor = gr.ImageEditor(
                    type="numpy", image_mode="RGBA", sources=(), transforms=(), layers=False,
                    canvas_size=(SIDE, SIDE), fixed_canvas=True, height=SIDE + 120, width=SIDE + 40,
                    brush=gr.Brush(default_size=16, colors=["#969696"], default_color="#969696", color_mode="fixed"),
                    label="Brush over the region to repair (grey); the eraser removes brushed areas")
                with gr.Row():
                    suggest_btn = gr.Button("Suggest suspicious areas (assistive)")
                    clear_btn = gr.Button("Clear mask")
                suggest_note = gr.Markdown("_Suggestions check 4 rules (unreachable rooms, dead-end walls, floating "
                                           "wall fragments, doors not between two spaces). They are hints, not a "
                                           "complete damage detector._")
        gr.Markdown("## 3 · Repair")
        with gr.Row():
            method = gr.Dropdown([METHOD_NAMES[m] for m in CFG["methods"]], value=METHOD_NAMES[CFG["methods"][0]],
                                 label="Method", scale=3)
            seed = gr.Number(0, precision=0, label="Seed (diffusion methods)", scale=1)
            go = gr.Button("Repair", variant="primary", scale=1)
            again = gr.Button("Generate another (seed + 1)", scale=1)
        with gr.Row():
            out_input = gr.Image(label="Input (grey = region to repair)", type="numpy", interactive=False)
            out_repaired = gr.Image(label="Repaired plan", type="numpy", interactive=False)
            out_diff = gr.Image(label="Difference: magenta = wall/door/window added, cyan = removed, "
                                      "amber = other change", type="numpy", interactive=False)
        report = gr.Markdown(elem_id="report")

        if gallery is not None:
            gallery.select(pick_sample, None, [state, mask_editor, status])
        start_current.click(lambda s: editor_start(s, False), state, editor)
        start_blank.click(lambda s: editor_start(s, True), state, editor)
        editor_btn.click(editor_use, editor, [state, mask_editor, status])
        upload_btn.click(upload_use, [upload, state], [state, mask_editor, status])
        suggest_btn.click(suggest_areas, state, [mask_editor, suggest_note])
        clear_btn.click(clear_mask, state, mask_editor)
        outputs = [out_input, out_repaired, out_diff, report]
        go.click(repair, [state, mask_editor, method, seed], outputs)
        again.click(repair_another, [state, mask_editor, method, seed], [seed] + outputs)
    return demo


REPAIRER = None    # set by setup(); module-level so the event handlers (and tests) can reach them
GALLERY = None


def setup():
    """Prepare the repairer (no checkpoint is read yet) and the sample gallery."""
    global REPAIRER, GALLERY
    REPAIRER = load_models()
    GALLERY = load_gallery()
    print(f"Running on {REPAIRER.device}; checkpoints load on first use.")


if __name__ == "__main__":
    setup()
    build_ui().queue().launch(server_name="127.0.0.1", server_port=CFG["server_port"], css=CSS)
