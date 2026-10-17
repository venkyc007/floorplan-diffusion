"""Parser building blocks: augmentations, crops, loss, metrics (no GPU, no dataset needed)."""

import numpy as np
import pytest
import torch

from src.stage2.data.classes16 import C16
from src.stage2.parse import augment
from src.stage2.parse.dataset import crop_around, pad_to_multiple, random_crop
from src.stage2.parse.losses import ParserLoss, class_weights, dice_loss
from src.stage2.parse.metrics import ParserMetrics, match_components
from src.stage2.parse.train import load_cfg, lr_factor
from src.utils import PROJECT_ROOT

CFG = load_cfg(PROJECT_ROOT / "configs" / "stage2" / "parser_v1.yaml", smoke=False)


def toy_drawing():
    img = np.full((300, 400), 255, np.uint8)
    lbl = np.full((300, 400), C16.index["background"], np.uint8)
    img[50:58, 50:350] = 0
    lbl[50:58, 50:350] = C16.index["wall"]
    lbl[58:250, 50:350] = C16.index["living"]
    lbl[50:58, 150:180] = C16.index["door"]
    img[50:58, 150:180] = 255
    return img, lbl


def test_random_crop_shapes_and_label_values():
    img, lbl = toy_drawing()
    rng = np.random.default_rng(0)
    for _ in range(20):
        x, y = random_crop(img, lbl, rng, CFG["data"], CFG["augment"], 128)
        assert x.shape == y.shape == (128, 128) and x.dtype == np.uint8
        assert set(np.unique(y)) <= set(np.unique(lbl)) | {C16.index["background"]}


def test_rare_centring_hits_the_door():
    img, lbl = toy_drawing()
    cfg = {**CFG["data"], "rare_centre_prob": 1.0}
    aug = {**CFG["augment"], "scale": [1.0, 1.0], "rot90": False, "flip": False}
    for k in ("binarise", "line_width", "blur", "noise", "jpeg", "contrast"):
        aug[k] = {**aug[k], "p": 0.0}
    x, y = random_crop(img, lbl, np.random.default_rng(1), cfg, aug, 64)
    assert y[32, 32] == C16.index["door"]


def test_crop_outside_is_paper_and_background():
    img, lbl = toy_drawing()
    ci, cl = crop_around(img, lbl, 0, 0, 40, 255)
    assert (ci[:20, :20] == 255).all() and (cl[:20, :20] == C16.index["background"]).all()


@pytest.mark.parametrize("fn,arg", [(augment.binarise, [-30, 30]), (augment.line_width, [1, 2]),
                                    (augment.blur, [0.4, 1.5]), (augment.noise, [3, 15]), (augment.jpeg, [30, 95])])
def test_photometric_keeps_shape(fn, arg):
    img, _ = toy_drawing()
    out = fn(img, np.random.default_rng(0), arg)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_binarise_is_black_and_white():
    img, _ = toy_drawing()
    img = augment.blur(img, np.random.default_rng(0), [1.0, 1.0])
    assert set(np.unique(augment.binarise(img, np.random.default_rng(0), [0, 0]))) <= {0, 255}


def test_class_weights_capped():
    w = class_weights(np.array([0.5, 0.0001] + [0.03] * 14), cap=10)
    assert float(w[0]) == pytest.approx(1.0) and float(w[1]) == 10.0


def test_loss_ignores_255_and_is_finite():
    torch.manual_seed(0)
    logits = torch.randn(2, 16, 32, 32, requires_grad=True)
    y = torch.randint(0, 16, (2, 32, 32))
    y[:, :8] = 255
    loss, parts = ParserLoss(torch.ones(16))(logits, y)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    perfect = torch.nn.functional.one_hot(y.clamp(max=15), 16).permute(0, 3, 1, 2).float() * 50
    assert float(dice_loss(perfect, y)) < 1e-3


def test_component_matching():
    gt = np.zeros((20, 20), bool)
    gt[2:5, 2:10] = True
    gt[12:15, 2:10] = True
    pred = np.zeros_like(gt)
    pred[2:5, 2:10] = True          # exact
    pred[12:15, 8:18] = True        # IoU < 0.5
    assert match_components(pred, gt) == (1, 1, 1)


def test_metrics_perfect():
    _, lbl = toy_drawing()
    m = ParserMetrics(["door", "window"])
    m.update(lbl, lbl)
    r = m.result()
    assert r["mIoU"] == pytest.approx(1.0) and r["F1_door"] == pytest.approx(1.0)


def test_pad_to_multiple():
    img, lbl = toy_drawing()
    pi, pl = pad_to_multiple(img, lbl, 32, 255, 255)
    assert pi.shape == (320, 416) and (pl[300:] == 255).all()


def test_lr_schedule():
    assert lr_factor(0, 0.0, 500) == pytest.approx(1 / 500)
    assert lr_factor(600, 0.0, 500) == pytest.approx(1.0)
    assert lr_factor(600, 1.0, 500) == pytest.approx(0.0)


def test_predict_safe_tiles_on_oom():
    """A model that 'runs out of memory' on big inputs gives the same labels through the tiles."""
    from src.stage2.parse.train import OOM_COUNT, predict_safe

    class PointModel(torch.nn.Module):
        def forward(self, x):
            if x.shape[-1] > 256:
                raise torch.OutOfMemoryError("fake")
            return torch.cat([x, 1 - x], 1)            # class 0 where bright, 1 where dark
    x = (torch.rand(1, 1, 320, 640) > 0.5).float()
    before = OOM_COUNT[0]
    out = predict_safe(PointModel(), x, amp=False, tile=256, overlap=32)
    assert OOM_COUNT[0] == before + 1
    assert np.array_equal(out, (x[0, 0] < 0.5).long().numpy())


def test_crop_rejects_empty_crops():
    img, lbl = toy_drawing()
    big_i = np.full((900, 900), 255, np.uint8)
    big_l = np.full((900, 900), C16.index["background"], np.uint8)
    big_i[:300, :400], big_l[:300, :400] = img, lbl
    cfg = {**CFG["data"], "rare_centre_prob": 0.0}
    rng = np.random.default_rng(0)
    fg = [(random_crop(big_i, big_l, rng, cfg, CFG["augment"], 128)[1] != C16.index["background"]).mean()
          for _ in range(30)]
    assert np.mean(np.array(fg) >= 0.05) > 0.5     # without rejection almost every crop would be empty


def test_whiten_padding_only_removes_border_rectangles():
    from src.stage2.data.rasterise import whiten_padding
    img, lbl = toy_drawing()
    img = img.copy()
    img[:, :40] = 0                                    # black padding strip at the left border
    out, n = whiten_padding(img, lbl)
    assert n == 300 * 40 and (out[:, :40] == 255).all()
    assert (out[50:58, 50:150] == 0).all()             # the black wall is kept


def test_fill_fixtures_gives_room_type():
    from src.stage2.data.plan_stats import fill_fixtures
    lab = np.full((20, 20), C16.index["bathroom"], np.uint8)
    lab[5:9, 5:9] = C16.index["toilet"]
    assert (fill_fixtures(lab) == C16.index["bathroom"]).all()
