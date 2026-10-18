"""CPU smoke tests of the FILL sampler: hybrid DDIM, x0 guidance, best-of-N verifier (tiny model)."""

import copy

import numpy as np
import torch
import yaml

from src.stage2.complete import sampler
from src.stage2.complete.gates34 import TINY
from src.stage2.complete.models16 import Diffusion16
from src.stage2.complete.windows import model_inputs
from src.stage2.constraints import toys
from src.stage2.constraints.constraints import one_hot
from src.utils import PROJECT_ROOT


def tiny_model():
    with open(PROJECT_ROOT / "configs" / "stage2" / "train16_constrained.yaml", encoding="utf-8") as f:
        m = yaml.safe_load(f)["model"]
    torch.manual_seed(0)
    return Diffusion16(copy.deepcopy(m) | TINY).eval()


def toy_batch(b=2):
    lab = np.stack([toys.valid_plan()[:64, :64]] * b)
    hole = np.zeros_like(lab)
    hole[:, 40:56, 10:30] = 1
    return model_inputs(lab, hole)


def test_hybrid_sample_keeps_known_region_and_shapes():
    model = tiny_model()
    cond, target, mask = toy_batch()
    sketch = target.argmax(1)
    g = torch.Generator().manual_seed(0)
    out = sampler.hybrid_sample(model, cond, sketch, t0=600, steps=5, generator=g)
    assert out.shape == (2, 64, 64)
    known = mask[:, 0] == 0
    assert torch.equal(out[known], target.argmax(1)[known])


def test_guidance_changes_the_sample_and_lowers_constraint_loss():
    model = tiny_model()
    cond, target, mask = toy_batch(1)
    a = sampler.hybrid_sample(model, cond, None, steps=5, generator=torch.Generator().manual_seed(1))
    b = sampler.hybrid_sample(model, cond, None, steps=5, eta_guidance=50.0, generator=torch.Generator().manual_seed(1))
    assert not torch.equal(a, b)
    # one guidance step on a noisy x0 lowers the weighted constraint loss
    w = sampler.guidance_weights()
    x0 = ((one_hot(target.argmax(1)) * 2 - 1) * 0.3 + torch.randn(1, 16, 64, 64) * 0.5).clamp(-1, 1)
    from src.stage2.constraints.constraints import CONST, Ctx, constraint_loss
    ctx = Ctx(hole=mask, known=cond[:, :16], const=CONST)

    def total(x):
        P = torch.softmax(x / 0.2, 1)
        return float(sum(v * constraint_loss(n, P, ctx, "absolute").sum() for n, v in w.items()))
    assert total(sampler.guide_x0(x0, mask, cond[:, :16], w, eta=0.05)) < total(x0)


def test_verifier_prefers_fewer_violations_then_sketch_agreement():
    cond, target, mask = toy_batch(1)
    good = target.argmax(1)[0]
    bad = good.clone()
    bad[44:46, 12:28] = 0                                   # a stray wall bar in the middle of the hole
    samples = torch.stack([bad, good, good.clone()])
    idx, counts, agree = sampler.best_of_n(samples, mask, cond[:, :16], sketch=good)
    assert counts[0] > counts[1] and idx in (1, 2)


def test_fill_runs_end_to_end():
    model = tiny_model()
    cond, target, mask = toy_batch(1)
    labels, info = sampler.fill(model, cond, sketch=target.argmax(1), n=2, steps=5, eta_guidance=0.1,
                                generator=torch.Generator().manual_seed(0))
    assert labels.shape == (1, 64, 64) and len(info[0]["violations"]) == 2
