"""Gradio app: Stage 1 and Stage 2 of the project in one page (switch with the tabs at the top).

    python app/app.py            (then open http://127.0.0.1:7860)

Stage 1 tab (ResPlan plans, 128 x 128, 6 classes; the region to repair is GIVEN):
  1. choose a plan: a val-benchmark sample, the plan editor, or an upload (PNG in our class colours / .npy);
  2. mark the region to repair: brush it with the mouse, or "Random mask" (the benchmark's own corruption rules);
  3. pick a method and repair -> input | repaired plan | difference (+ scores when the true plan is known).
Stage 2 tab (real black-and-white drawings, 16 classes; the system must FIND what is missing):
  1. choose a drawing: a held-out val drawing with one part erased (gallery), or upload ANY floor plan image;
  2a. automatic: PARSE -> CHECK -> FILL -> VERIFY, every step shown (parsed map, what was flagged, filled plan,
      additions in red on the drawing);
  2b. manual: if CHECK misses something, brush the missing region yourself -> only PARSE + FILL run.
All model work happens in src/stage1/interface/ (Stage 1) and src/stage2/interface/ (Stage 2); this file only builds the
page. Settings: configs/stage1/app.yaml (Stage 1, frozen Day 7 selection) and configs/stage2/app.yaml (Stage 2 tab; the
pipeline's frozen settings are in configs/stage2/pipeline.yaml).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                    # so `src` imports work with "python app/app.py"

import cv2  # noqa: E402
import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.stage1.data.classes import BACKGROUND, CLASSES, COLORS, WALL  # noqa: E402
from src.stage1.data.corruptions import UNKNOWN  # noqa: E402
from src.stage1.interface.inference import (METHOD_NAMES, STOCHASTIC, CheckpointMissing, Repairer,  # noqa: E402
                                     describe, load_config)
from src.stage1.interface.plan_io import InputError, load_file  # noqa: E402
from src.stage1.interface.random_mask import random_corruption  # noqa: E402
from src.stage1.interface.render import (difference_image, editor_to_labels, layers_to_mask,  # noqa: E402
                                  mask_layer, plan_image)
from src.stage1.interface.report import gt_scores, rule_table, to_markdown  # noqa: E402
from src.stage1.interface.suggest import suggest  # noqa: E402
from src.stage2.interface import demo as S2  # noqa: E402
from src.utils import project_path  # noqa: E402

CFG = load_config()
SCALE = CFG["display_scale"]
SIDE = 128 * SCALE
THRESHOLDS = yaml.safe_load(open(project_path("configs/stage1/arch_metrics.yaml"), encoding="utf-8"))
LABEL_TO_METHOD = {METHOD_NAMES[m]: m for m in CFG["methods"]}
EDIT2 = 760                       # Stage 2 mask canvas (px, square); the drawing is fitted into its top-left
# Stage 1 plans are tiny pixel maps: show them with sharp pixels instead of browser smoothing.
CSS = ".pixelated img, .pixelated canvas { image-rendering: pixelated; }"


def hex_color(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# ================================================================== Stage 1
def load_models():
    """Check that both Stage 1 checkpoints exist (they are loaded later, on first use)."""
    try:
        return Repairer(CFG)
    except CheckpointMissing as e:
        print(f"\nERROR: {e}\n", file=sys.stderr)
        sys.exit(1)


def load_gallery():
    path = project_path(CFG["gallery"])
    if not path.exists():
        print(f"Note: no Stage 1 gallery at {path} - make it with: python -m src.stage1.interface.make_gallery")
        return None
    g = np.load(path)
    return {k: g[k] for k in g.files}


def new_state(plan, gt=None, source="", preset_mask=None):
    return {"plan": plan, "gt": gt, "source": source, "preset_mask": preset_mask, "random_seed": 0}


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


def random_mask(state):
    """Remove one random part of the COMPLETE plan with the benchmark's rules; that plan becomes the truth."""
    if state is None:
        raise gr.Error("Choose a plan first (step 1).")
    complete = state["gt"] if state["gt"] is not None else state["plan"]
    if (complete == UNKNOWN).any():
        raise gr.Error("This plan has unknown (grey) pixels and no true plan, so there is no complete plan to "
                       "remove a part from. Pick a val sample or draw a complete plan.")
    seed = state.get("random_seed", 0) + 1
    out = random_corruption(complete, seed)
    if out is None:
        raise gr.Error("No door, window, wall segment, room or region fits this plan; brush a mask instead.")
    plan, mask, truth, text = out
    new = {**new_state(plan, truth, f"{state['source'].split(' · ')[0]} · {text}", mask), "random_seed": seed}
    return new, mask_editor_value(plan, mask), plan_status(new, "Press again for another random removal.")


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


def repair(state, mask_value, method_label, seed, progress=gr.Progress()):
    if state is None:
        raise gr.Error("Choose a plan first (step 1).")
    mask = current_mask(state, mask_value)
    if not mask.any():
        raise gr.Error("Mark the region to repair first (step 2): brush it, or press 'Random mask'.")
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


def legend_html():
    chips = "".join(f'<span style="display:inline-block;margin-right:12px"><span style="display:inline-block;'
                    f'width:14px;height:14px;background:{hex_color(COLORS[c])};border:1px solid #999;'
                    f'vertical-align:middle"></span> {CLASSES[c]}</span>' for c in range(len(CLASSES)))
    grey = ('<span style="display:inline-block"><span style="display:inline-block;width:14px;height:14px;'
            'background:#969696;border:1px solid #999;vertical-align:middle"></span> region to repair / unknown'
            '</span>')
    return f"<div>{chips}{grey}</div>"


def stage1_tab():
    brush_colors = [hex_color(COLORS[c]) for c in range(len(CLASSES))]
    state = gr.State(None)
    gr.Markdown("Plans from **ResPlan** as 128 × 128 maps with 6 classes. You say **where** to repair; the model "
                "redraws that region. **1** choose a plan → **2** mark the region → **3** repair.")
    gr.HTML(legend_html())
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1 · Plan")
            with gr.Tabs():
                with gr.Tab("Samples (val set)"):
                    if GALLERY is None:
                        gr.Markdown("No sample gallery found. Make it with `python -m src.stage1.interface.make_gallery`.")
                        gallery = None
                    else:
                        items = [(plan_image(p, 2, m), f"{t} · {s}") for p, m, t, s in
                                 zip(GALLERY["inputs"], GALLERY["masks"], GALLERY["types"], GALLERY["size_bins"])]
                        gallery = gr.Gallery(items, columns=4, height=520, allow_preview=False,
                                             label="24 val-benchmark cases (caption: removed part · hole size)",
                                             elem_classes="pixelated")
                with gr.Tab("Editor"):
                    gr.Markdown("Draw with the class colours; to **delete** something, paint over it with room, "
                                "outdoor or background colour. Then press *Use edited plan*.")
                    with gr.Row():
                        start_current = gr.Button("Start from current plan", size="sm")
                        start_blank = gr.Button("Start blank", size="sm")
                    editor = gr.ImageEditor(
                        type="numpy", image_mode="RGB", sources=(), transforms=(), layers=False,
                        canvas_size=(SIDE, SIDE), fixed_canvas=True, height=SIDE + 120, width=SIDE + 40,
                        brush=gr.Brush(default_size=8, colors=brush_colors, default_color=brush_colors[WALL],
                                       color_mode="fixed"),
                        label="Plan editor (128 × 128 plan shown at 4×)", elem_classes="pixelated")
                    editor_btn = gr.Button("Use edited plan", variant="primary")
                with gr.Tab("Upload"):
                    gr.Markdown("A **PNG** in our 6 class colours (each pixel snaps to the nearest class colour; "
                                "grey = unknown) or a **.npy** label map (values 0–5 = " + ", ".join(CLASSES)
                                + "; 255 = unknown). The plan is framed on its walls, doors and windows and "
                                "resized to 128 × 128 (nearest neighbour), as in our data.")
                    upload = gr.File(file_types=[".png", ".npy", ".jpg", ".jpeg", ".bmp", ".webp"],
                                     type="filepath", label="Plan file")
                    upload_btn = gr.Button("Use uploaded file", variant="primary")
            status = gr.Markdown("Choose a plan to start.")
        with gr.Column():
            gr.Markdown("### 2 · Region to repair")
            mask_editor = gr.ImageEditor(
                type="numpy", image_mode="RGBA", sources=(), transforms=(), layers=False,
                canvas_size=(SIDE, SIDE), fixed_canvas=True, height=SIDE + 120, width=SIDE + 40,
                brush=gr.Brush(default_size=16, colors=["#969696"], default_color="#969696", color_mode="fixed"),
                label="Brush over the region to repair (grey); the eraser removes brushed areas",
                elem_classes="pixelated")
            with gr.Row():
                random_btn = gr.Button("Random mask (benchmark rules)")
                suggest_btn = gr.Button("Suggest suspicious areas")
                clear_btn = gr.Button("Clear mask")
            suggest_note = gr.Markdown("_Random mask removes one door, window, wall segment, room or region from the "
                                       "complete plan, exactly as the benchmarks were made. Suggestions check 4 "
                                       "rules; they are hints, not a complete damage detector._")
    gr.Markdown("### 3 · Repair")
    with gr.Row():
        method = gr.Dropdown([METHOD_NAMES[m] for m in CFG["methods"]], value=METHOD_NAMES[CFG["methods"][0]],
                             label="Method (first = our final method)", scale=3)
        seed = gr.Number(0, precision=0, label="Seed (diffusion methods)", scale=1)
        go = gr.Button("Repair", variant="primary", scale=1)
        again = gr.Button("Generate another (seed + 1)", scale=1)
    with gr.Row():
        out_input = gr.Image(label="Input (grey = region to repair)", type="numpy", interactive=False,
                             elem_classes="pixelated")
        out_repaired = gr.Image(label="Repaired plan", type="numpy", interactive=False, elem_classes="pixelated")
        out_diff = gr.Image(label="Difference: magenta = wall/door/window added, cyan = removed, amber = other change",
                            type="numpy", interactive=False, elem_classes="pixelated")
    report = gr.Markdown()

    if gallery is not None:
        gallery.select(pick_sample, None, [state, mask_editor, status])
    start_current.click(lambda s: editor_start(s, False), state, editor)
    start_blank.click(lambda s: editor_start(s, True), state, editor)
    editor_btn.click(editor_use, editor, [state, mask_editor, status])
    upload_btn.click(upload_use, [upload, state], [state, mask_editor, status])
    random_btn.click(random_mask, state, [state, mask_editor, status])
    suggest_btn.click(suggest_areas, state, [mask_editor, suggest_note])
    clear_btn.click(clear_mask, state, mask_editor)
    outputs = [out_input, out_repaired, out_diff, report]
    go.click(repair, [state, mask_editor, method, seed], outputs)
    again.click(repair_another, [state, mask_editor, method, seed], [seed] + outputs)


# ================================================================== Stage 2
def s2_state(grey, case, source):
    return {"grey": grey, "case": case, "source": source}


def s2_canvas(grey):
    """Square white canvas with the drawing fitted into its top-left corner -> (RGB canvas, scale)."""
    h, w = grey.shape
    f = EDIT2 / max(h, w)
    small = cv2.resize(grey, (max(1, round(w * f)), max(1, round(h * f))),
                       interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_NEAREST)
    canvas = np.full((EDIT2, EDIT2, 3), 255, np.uint8)
    canvas[:small.shape[0], :small.shape[1]] = small[..., None]
    return canvas, f


def s2_editor_value(grey):
    canvas, _ = s2_canvas(grey)
    return {"background": canvas, "layers": [], "composite": None}


def s2_status(st):
    text = f"✅ **Current drawing:** {st['source']} ({st['grey'].shape[1]} × {st['grey'].shape[0]} px)"
    if st["case"] is not None:
        text += " · the erased part is known, so the result is scored against the truth"
    return text


def s2_pick(evt: gr.SelectData):
    case = GALLERY2[evt.index]
    grey, original, _, _ = S2.load_case(case)
    st = s2_state(grey, case, f"held-out val drawing {case['plan_id']}, {S2.case_caption(case)}")
    label = None if case["box"] is None else f"erased: {case['kind']}"
    answer = S2.box_on_drawing(original, case["box"], label=label)
    return st, S2.fit(cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)), s2_editor_value(grey), s2_status(st), answer


def s2_upload(image):
    if image is None:
        raise gr.Error("Upload an image of a floor plan first.")
    try:
        grey = S2.to_grey(image)
    except S2.InputError as e:
        raise gr.Error(f"Image not usable: {e}")
    st = s2_state(grey, None, "your uploaded drawing (no ground truth: the result is not scored)")
    return (st, S2.fit(cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)), s2_editor_value(grey), s2_status(st),
            None)


def s2_report(steps, st, mode):
    res, t = steps["res"], steps["timings"]
    lines = [f"**{mode}** · parse {t.get('parse', 0)} s · check {t.get('detect', 0)} s · fill {t.get('fill', 0)} s "
             f"· parse scale {res['parse_info']['parse_scale']:.2f}"]
    if st["case"] is not None:
        lines.append(S2.score_markdown(S2.score(res, st["case"]), st["case"]))
    lines.append("**What the system reports:**")
    lines += [f"- {m}" for m in steps["messages"]] or ["- nothing flagged: the system thinks the drawing is complete"]
    return "\n".join(lines)


def s2_auto(st, seed, fast):
    if st is None:
        raise gr.Error("Choose a drawing first (step 1).")
    steps = S2.run(st["grey"], None, int(seed), bool(fast))
    return (steps["parsed"], steps["flags"], steps["completed"], steps["overlay"],
            s2_report(steps, st, "Automatic: PARSE → CHECK → FILL → VERIFY"), steps["rows"])


def s2_manual(st, value, seed, fast):
    if st is None:
        raise gr.Error("Choose a drawing first (step 1).")
    layers = (value or {}).get("layers") if isinstance(value, dict) else None
    if not layers:
        raise gr.Error("Brush over the missing region first (orange).")
    alpha = np.zeros((EDIT2, EDIT2), bool)
    for layer in layers:
        a = np.asarray(layer)
        if a.ndim == 3 and a.shape[2] == 4:
            alpha |= a[..., 3] > 0
    h, w = st["grey"].shape
    _, f = s2_canvas(st["grey"])
    crop = alpha[:max(1, round(h * f)), :max(1, round(w * f))].astype(np.uint8)
    mask = cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    boxes = S2.mask_boxes(mask)
    if not boxes:
        raise gr.Error("The brushed region is too small; brush over the whole missing area.")
    steps = S2.run(st["grey"], boxes, int(seed), bool(fast))
    return (steps["parsed"], steps["flags"], steps["completed"], steps["overlay"],
            s2_report(steps, st, f"Manual: your {len(boxes)} region(s) → PARSE → FILL (CHECK skipped)"),
            steps["rows"])


def s2_clear(st):
    if st is None:
        raise gr.Error("Choose a drawing first (step 1).")
    return s2_editor_value(st["grey"])


def stage2_tab():
    st = gr.State(None)
    gr.Markdown("A real black-and-white floor plan **drawing**, any size or line width. The system must **find** what "
                "is missing and draw it: **PARSE** (our segmentation network reads 16 classes) → **CHECK** (rules + "
                "learned detectors flag missing parts) → **FILL** (16-class diffusion inpainting per hole) → "
                "**VERIFY** (best of N samples by fewest rule violations). If CHECK misses something, mark it "
                "yourself in 2b: then only PARSE and FILL run. _About 1–2 minutes per drawing on a laptop CPU._")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1 · Drawing")
            with gr.Tabs():
                with gr.Tab("Examples (held-out val)"):
                    if not GALLERY2:
                        gr.Markdown("No examples yet. Make them with `python -m src.stage2.interface.make_gallery`.")
                        gallery = None
                    else:
                        gallery = gr.Gallery([(c["drawing"], S2.case_caption(c)) for c in GALLERY2], columns=3,
                                             height=520, allow_preview=False,
                                             label="18 val drawings in fixed order: one part erased each (2 intact)")
                with gr.Tab("Upload any drawing"):
                    gr.Markdown("Any floor plan image (PNG / JPG; scanned, exported or a photo of a print). Colour "
                                "is turned to grey; white-on-black drawings are inverted. Erase something first to "
                                "see whether the system finds it.")
                    upload = gr.Image(type="numpy", sources=["upload", "clipboard"], label="Floor plan image")
                    upload_btn = gr.Button("Use this drawing", variant="primary")
            status = gr.Markdown("Choose a drawing to start.")
            with gr.Accordion("Show the answer (where the part was erased)", open=False):
                answer = gr.Image(label="Original drawing, erased part in orange", type="numpy", interactive=False)
        with gr.Column(scale=1):
            gr.Markdown("### 2 · Find and fix")
            current = gr.Image(label="Input drawing", type="numpy", interactive=False)
            with gr.Row():
                seed = gr.Number(0, precision=0, label="Seed", scale=1)
                fast = gr.Checkbox(value=S2.CFG["fast"], label="Fast settings (N = 2, 15 steps; off = "
                                                               "evaluation settings, slower)", scale=3)
            auto_btn = gr.Button("2a · Automatic: find and fix what is missing", variant="primary")
            with gr.Accordion("2b · Manual: mark the missing region yourself", open=False):
                editor = gr.ImageEditor(type="numpy", image_mode="RGBA", sources=(), transforms=(), layers=False,
                                        canvas_size=(EDIT2, EDIT2), fixed_canvas=True, height=EDIT2 // 1.4 + 120,
                                        brush=gr.Brush(default_size=24, colors=["#ff8c00"], default_color="#ff8c00",
                                                       color_mode="fixed"),
                                        label="Brush over the missing region (orange); each separate region is "
                                              "filled on its own")
                with gr.Row():
                    manual_btn = gr.Button("Fill the marked region(s)", variant="primary")
                    clear_btn = gr.Button("Clear marks")
    gr.Markdown("### 3 · Every step")
    gr.HTML(S2.legend_html())
    with gr.Row():
        o_parse = gr.Image(label="PARSE: what the parser read (16-class map)", type="numpy", interactive=False)
        o_flags = gr.Image(label="CHECK: what was flagged as missing (red) / your boxes (blue)", type="numpy",
                           interactive=False)
    with gr.Row():
        o_fill = gr.Image(label="FILL + VERIFY: completed plan (changes outlined in magenta)", type="numpy",
                          interactive=False)
        o_overlay = gr.Image(label="Result on your drawing: additions in red (doors as gap + leaf + arc)",
                             type="numpy", interactive=False)
    o_report = gr.Markdown()
    o_rows = gr.Dataframe(headers=S2.HOLE_HEADERS, label="Holes: what was flagged, by which rule or detector, and "
                                                        "what the verifier counted", wrap=True)

    if gallery is not None:
        gallery.select(s2_pick, None, [st, current, editor, status, answer])
    upload_btn.click(s2_upload, upload, [st, current, editor, status, answer])
    outs = [o_parse, o_flags, o_fill, o_overlay, o_report, o_rows]
    auto_btn.click(s2_auto, [st, seed, fast], outs)
    manual_btn.click(s2_manual, [st, editor, seed, fast], outs)
    clear_btn.click(s2_clear, st, editor)


# ================================================================== page
def build_ui():
    about = (ROOT / "app" / "ABOUT.md").read_text(encoding="utf-8") if (ROOT / "app" / "ABOUT.md").exists() else ""
    with gr.Blocks(title="Floor plan completion · IE643") as page:
        gr.Markdown("# Diffusion-based reconstruction of missing floor plan components\n"
                    "IE 643 course project · switch between the two stages with the tabs below. "
                    f"Running on **{REPAIRER.device.type.upper()}**.")
        with gr.Tabs():
            with gr.Tab("Stage 1 · plan + given region (ResPlan)"):
                stage1_tab()
            with gr.Tab("Stage 2 · real drawings: find + fill (CubiCasa5K)"):
                stage2_tab()
            if about:
                with gr.Tab("How it works · results"):
                    gr.Markdown(about)
    return page


REPAIRER = None    # set by setup(); module-level so the event handlers (and tests) can reach them
GALLERY = None
GALLERY2 = []


def setup():
    """Prepare the Stage 1 repairer and both galleries (no checkpoint is read yet: models load on first use)."""
    global REPAIRER, GALLERY, GALLERY2
    REPAIRER = load_models()
    GALLERY = load_gallery()
    GALLERY2 = S2.load_gallery()
    print(f"Running on {REPAIRER.device}; checkpoints load on first use. Stage 2 examples: {len(GALLERY2)}")


if __name__ == "__main__":
    setup()
    build_ui().queue().launch(server_name="127.0.0.1", server_port=CFG["server_port"], css=CSS)
