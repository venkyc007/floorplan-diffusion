"""Fast checks for src/models and the training helpers. Run:  python -m pytest tests/"""

import torch

from src.models import build_model
from src.train import lr_factor

TINY = {"block_out_channels": [8, 16, 16, 16], "layers_per_block": 1, "attention_from_level": 3,
        "attention_head_dim": 8, "norm_num_groups": 4, "mask_weight": 5.0,
        "num_train_timesteps": 1000, "beta_schedule": "squaredcos_cap_v2", "ddim_steps": 3}


def toy_batch(b=2, size=64):
    """Known plan: class 3 (room) everywhere, a square of class 0 (wall);
    the mask covers the square."""
    labels = torch.full((b, size, size), 3)
    labels[:, 20:30, 20:30] = 0
    mask = torch.zeros(b, 1, size, size)
    mask[:, :, 18:32, 18:32] = 1
    onehot = torch.nn.functional.one_hot(labels, 6).permute(0, 3, 1, 2).float()
    cond = torch.cat([onehot * (1 - mask), mask], dim=1)
    return cond, onehot, mask, labels


def test_diffusion_keeps_known_region():
    """Sampling must return the known plan exactly outside the mask."""
    torch.manual_seed(0)
    model = build_model({**TINY, "kind": "diffusion"}).eval()
    cond, _, mask, labels = toy_batch()
    pred = model.predict(cond, steps=3, generator=torch.Generator().manual_seed(0))
    outside = mask[:, 0] == 0
    assert pred.shape == labels.shape
    assert (pred[outside] == labels[outside]).all()


def test_losses_are_finite_and_backpropagate():
    cond, target, mask, _ = toy_batch()
    for kind in ("diffusion", "unet"):
        model = build_model({**TINY, "kind": kind})
        loss = model.loss(cond, target, mask)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_unet_predicts_label_maps():
    model = build_model({**TINY, "kind": "unet"}).eval()
    cond, _, _, labels = toy_batch()
    pred = model.predict(cond)
    assert pred.shape == labels.shape and pred.min() >= 0 and pred.max() < 6


def test_learning_rate_schedule():
    """Linear warm-up to 1, then cosine down to 0."""
    assert lr_factor(0, 10, 100) == 0.1
    assert lr_factor(9, 10, 100) == 1.0
    assert abs(lr_factor(55, 10, 100) - 0.5) < 1e-9
    assert lr_factor(100, 10, 100) == 0.0
