"""Unit tests for the interface back-end (src/interface/). No models or data needed.
Run:  python -m pytest tests/"""

import io

import numpy as np
import pytest
from PIL import Image

from src.data.classes import BACKGROUND, ROOM, WALL
from src.data.corruptions import UNKNOWN
from src.eval.figures import colorize
from src.interface.inference import size_bin
from src.interface.plan_io import (InputError, frame_on_building, load_npy, load_png, snap_colors,
                                   structure_box)
from src.interface.render import difference_image, layers_to_mask, plan_image
from src.interface.report import rule_table
from src.interface.suggest import suggest
from tests.test_metrics import toy_plan

CFG = {"max_side_px": 4096, "max_off_palette_pct": 10, "off_palette_distance": 60, "unknown_distance": 12,
       "min_wall_px": 100, "keep_frame_long_side": [112, 124]}
TH = {"min_area": 72.0, "min_width": 7.64}


def png_bytes(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def npy_bytes(arr, allow_pickle=False):
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=allow_pickle)
    return buf.getvalue()


def framed_toy():
    return frame_on_building(toy_plan())


def test_framing_scales_the_structure_to_120_px():
    plan = framed_toy()
    r0, r1, c0, c1 = structure_box(plan)
    assert plan.shape == (128, 128)
    assert max(r1 - r0, c1 - c0) + 1 in (119, 120, 121)
    assert abs((r0 + r1) / 2 - 63.5) <= 1 and abs((c0 + c1) / 2 - 63.5) <= 1     # centred


def test_colour_snapping_survives_noise_and_grey_is_unknown():
    plan = toy_plan()
    rgb = colorize(plan).astype(int) + np.random.default_rng(0).integers(-15, 16, (40, 40, 3))
    labels, _ = snap_colors(np.clip(rgb, 0, 255))
    assert (labels == plan).all()
    labels, _ = snap_colors(np.full((4, 4, 3), 150))
    assert (labels == UNKNOWN).all()


def test_png_round_trip_framed_and_upscaled():
    plan = framed_toy()
    assert (load_png(png_bytes(colorize(plan)), CFG) == plan).all()          # already framed: unchanged
    big = np.repeat(np.repeat(colorize(plan), 4, 0), 4, 1)                    # a 512-px screenshot of it
    assert (load_png(png_bytes(big), CFG) == plan).mean() > 0.99


def test_npy_label_map_and_one_hot():
    plan = framed_toy()
    assert (load_npy(npy_bytes(plan), CFG) == plan).all()
    onehot = (np.arange(6)[:, None, None] == plan[None]).astype(np.float32)
    assert (load_npy(npy_bytes(onehot), CFG) == plan).all()


@pytest.mark.parametrize("data, loader", [
    (b"not an image at all", load_png),
    (png_bytes(np.random.default_rng(0).integers(0, 256, (64, 64, 3))), load_png),   # a photo-like image
    (png_bytes(np.full((64, 64, 3), 255)), load_png),                                # blank: no walls
    (npy_bytes(np.full((64, 64), 9)), load_npy),                                      # unknown class ids
    (npy_bytes(np.zeros((2, 3, 64, 64))), load_npy),                                  # wrong shape
    (npy_bytes(np.array([{"x": 1}], dtype=object), allow_pickle=True), load_npy),     # pickled object
    (b"garbage", load_npy),
], ids=["png-garbage", "png-photo", "png-blank", "npy-bad-ids", "npy-bad-shape", "npy-pickle", "npy-garbage"])
def test_malformed_input_is_refused_with_input_error(data, loader):
    with pytest.raises(InputError):
        loader(data, CFG)


def test_mask_from_brush_layer():
    layer = np.zeros((512, 512, 4), np.uint8)
    layer[40:80, 100:140, 3] = 255                     # painted screen pixels
    mask = layers_to_mask([layer], 4)
    assert mask.shape == (128, 128) and mask[10:20, 25:35].all() and mask.sum() == 100


def test_suggestions_find_a_floating_fragment_only_when_present():
    plan = toy_plan()
    mask, found = suggest(plan, TH)
    assert found == [] and not mask.any()
    plan[26:28, 10:12] = WALL                           # floating 4-px wall blob
    mask, found = suggest(plan, TH)
    names = {f[0] for f in found}            # the blob's ends also count as dead-end walls
    assert "floating wall fragment" in names and names <= {"floating wall fragment", "dead-end wall"} and mask[26, 10]


def test_size_bin_uses_the_completed_building():
    completed = np.full((100, 100), BACKGROUND, np.uint8)
    completed[:50, :] = ROOM                             # building = 5,000 px
    mask = np.zeros((100, 100), bool)
    mask[:10, :10] = True                                # 2% -> small
    assert size_bin(mask, completed, [2.7, 7.7])[0] == "small"
    mask[:20, :20] = True                                # 8% -> large
    assert size_bin(mask, completed, [2.7, 7.7])[0] == "large"


def test_rule_table_and_pictures():
    gt = toy_plan()
    mask = np.zeros((40, 40), bool)
    mask[22:32, 6:18] = True
    bad = gt.copy()
    bad[26:28, 10:12] = WALL
    rows = rule_table(bad, mask, TH, gt)
    failed = {name for name, _, _, ok in rows if not ok}
    assert len(rows) == 6 and failed == {"no floating wall fragments", "no dead-end walls"}   # the blob's ends dead-end too
    before = gt.copy()
    before[mask] = UNKNOWN
    diff = difference_image(before, bad, mask, 2)
    assert diff.shape == (80, 80, 3) and (diff == (255, 0, 255)).all(-1).any()   # drawn wall highlighted
    assert plan_image(gt, 4, mask).shape == (160, 160, 3)


def test_missing_checkpoint_gives_a_clear_message():
    from src.interface.inference import CheckpointMissing, check_checkpoints
    with pytest.raises(CheckpointMissing, match=r"unet_matched: .*does_not_exist.pt"):
        check_checkpoints({"checkpoints": {"unet_matched": "outputs/does_not_exist.pt"}})
