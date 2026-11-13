"""Novelty models: the Fourier blocks start as the identity and the data windows have the promised layout."""

import numpy as np
import torch

from src.novelty.data import fill_window, hole_box, locate_canvas, to_model
from src.novelty.ffc import SpectralBlock
from src.novelty.models import build_large
from src.stage2.complete.gates34 import TINY
from src.stage2.complete.models16 import build16
from src.stage2.data.classes16 import C16

BASE = {"kind": "unet", "attention_from_level": 3, "num_train_timesteps": 1000, "beta_schedule": "squaredcos_cap_v2",
        "ddim_steps": 50, "mask_weight": 5.0} | TINY


def test_spectral_block_starts_as_identity():
    x = torch.randn(2, 32, 16, 16)
    assert torch.equal(SpectralBlock(32)(x), x)


def test_warm_started_model_is_unchanged_by_new_blocks():
    torch.manual_seed(0)
    plain = build16(BASE).eval()
    ffc = build_large(BASE | {"ffc": {"enabled": True, "at": ["down2", "mid", "up0"]}}).eval()
    missing, unexpected = ffc.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("spectral.") for k in missing) and missing
    cond = torch.rand(1, C16.n + 1, 64, 64)
    with torch.no_grad():
        assert torch.allclose(plain(cond), ffc(cond), atol=1e-6)
    # once the gate is opened the blocks do change the output (the hooks are live)
    for b in ffc.spectral.blocks.values():
        torch.nn.init.normal_(b.out.weight, std=0.1)
    with torch.no_grad():
        assert not torch.allclose(plain(cond), ffc(cond), atol=1e-4)


def test_fill_window_keeps_context_and_puts_truth_in_the_hole():
    rng = np.random.default_rng(0)
    parsed = rng.integers(0, C16.n, (120, 150)).astype(np.uint8)
    gt = rng.integers(0, C16.n, (120, 150)).astype(np.uint8)
    area = np.zeros(parsed.shape, bool)
    area[40:80, 50:100] = True
    ctx, tgt, hole = fill_window(parsed, gt, area, rng, win=128)
    assert ctx.shape == tgt.shape == hole.shape == (128, 128)
    assert (tgt[hole == 0] == ctx[hole == 0]).all()                     # outside the hole: the parser's map
    y0, x0, y1, x1 = hole_box(area, parsed.shape)
    assert hole.sum() >= 0.8 * (y1 - y0 + 1) * (x1 - x0 + 1)             # the (jittered) box of the area
    cond, target, mask = to_model(ctx, tgt, hole)
    assert cond.shape == (C16.n + 1, 128, 128) and target.shape == (C16.n, 128, 128)
    assert float((cond[:C16.n] * mask).abs().sum()) == 0                # nothing of the answer inside the hole
    assert fill_window(parsed, gt, np.ones(parsed.shape, bool), rng, win=128) is None   # too big for the window


def test_locate_canvas_shapes():
    rng = np.random.default_rng(1)
    for shape in ((90, 140), (400, 500)):
        parsed = np.zeros(shape, np.uint8)
        area = np.zeros(shape, bool)
        area[30:60, 40:90] = True
        lab, tgt = locate_canvas(parsed, area, rng, canvas=320)
        assert lab.shape == tgt.shape == (320, 320) and tgt.any()


def test_edit_mode_keeps_the_parsed_map_visible_in_the_hole():
    rng = np.random.default_rng(0)
    parsed = rng.integers(0, C16.n, (120, 150)).astype(np.uint8)
    gt = rng.integers(0, C16.n, (120, 150)).astype(np.uint8)
    area = np.zeros(parsed.shape, bool)
    area[40:80, 50:100] = True
    ctx, tgt, hole = fill_window(parsed, gt, area, rng, win=128, box=(38, 48, 82, 104))   # an explicit (locator) box
    assert hole.sum() == 45 * 57
    cond_edit, target, mask = to_model(ctx, tgt, hole, hide=False)
    cond_hide, _, _ = to_model(ctx, tgt, hole, hide=True)
    assert float((cond_edit[:C16.n] * mask).sum()) == float(mask.sum())      # one class per hole pixel: the parsed map
    assert torch.equal(cond_edit[:C16.n] * (1 - mask), cond_hide[:C16.n])    # identical outside the hole
    assert torch.equal(cond_edit[:C16.n].argmax(0), torch.from_numpy(ctx.astype(np.int64)))
