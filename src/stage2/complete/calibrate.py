"""Constraint weights by GRADIENT SIZE, not loss value (spec section 3; fixes the Stage 1 bug).

    python -m src.stage2.complete.calibrate --zip completion_train.zip --stage1 <best.pt> \
        --config configs/stage2/train16_constrained.yaml [--batches 50] [--batch 8] [--write]

  1. warm-start the 16-class diffusion from the Stage 1 checkpoint (class expansion);
  2. for each of `batches` batches of real windows: draw t, predict x0 (no parameter
     gradients), then G_0 = || dL_rec / dx0 || and G_i = || d(g(t) * L_i) / dx0 || for every
     active constraint (its mode from the config, as in training);
  3. w_i = rho * median(G_0) / median(G_i), rho = 0.05, or 0.4 / n if n * 0.05 > 0.4;
  4. --write: put the weights into BOTH train16 configs (identical except the switch) and
     write outputs/stage2/constraint_weights.md with every G_i.
Batches where a constraint has no signal (G_i = 0, e.g. no qualifying room for C17) are left
out of its median; a constraint with no signal at all gets weight 0 and is flagged.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.stage2.complete.data16 import fixed_windows, load_plans
from src.stage2.complete.models16 import gate
from src.stage2.complete.warm_start import warm_start
from src.stage2.complete.windows import model_inputs
from src.stage2.constraints.constraints import Ctx, constraint_loss, load_constants
from src.utils import PROJECT_ROOT

T_MAX = 400
MIN_SIGNAL = 10        # constraints with signal in fewer batches get the median weight of the others
CAP = 10.0             # every weight <= CAP x the median weight
WEIGHTS_YAML = PROJECT_ROOT / "outputs" / "stage2" / "constraint_weights.yaml"
CONFIGS = [PROJECT_ROOT / "configs" / "stage2" / f"train16_{k}.yaml" for k in ("constrained", "control")]
OUT_MD = PROJECT_ROOT / "outputs" / "stage2" / "constraint_weights.md"


def grad_norm(loss, x):
    g, = torch.autograd.grad(loss, x, retain_graph=True, allow_unused=True)
    return 0.0 if g is None else float(g.norm())


def measure(model, labels, masks, ccfg, batches, batch, seed=0):
    """{name: [G per batch]} including 'rec'."""
    torch.manual_seed(seed)
    dev = next(model.parameters()).device
    const = load_constants()
    out = {"rec": []} | {n: [] for n in ccfg["active"]}
    model.eval()
    for b in range(batches):
        sl = slice(b * batch, (b + 1) * batch)
        cond, target, mask = (v.to(dev) for v in model_inputs(labels[sl], masks[sl]))
        x0 = target * 2 - 1
        t = torch.randint(0, T_MAX + 1, (len(x0),)).to(dev)     # user: t uniform in 0-400 (constraints fully on)
        x_t = model.train_scheduler.add_noise(x0, torch.randn_like(x0), t)
        with torch.no_grad():
            pred = model(x_t, cond, t)
        pred = pred.float().requires_grad_()
        weight = 1 + (model.mcfg["mask_weight"] - 1) * mask
        rec = (((pred - x0) ** 2) * weight).sum() / (weight.sum() * x0.shape[1])
        out["rec"].append(grad_norm(rec, pred))
        P = F.softmax(pred / ccfg["temperature"], dim=1)
        ctx = Ctx(hole=mask, Y=target, known=target * (1 - mask), const=const, K=ccfg.get("K", 96))
        g = gate(t, ccfg)
        for name in ccfg["active"]:
            loss = (constraint_loss(name, P, ctx, ccfg.get("modes", {}).get(name)) * g).mean()
            out[name].append(grad_norm(loss, pred) if loss.requires_grad else 0.0)
        print(f"batch {b + 1}/{batches}", flush=True)
    return out


def weights_from(G, rho=0.05, cap=0.4):
    """Weights + notes. Median of G_i over batches with signal; < MIN_SIGNAL batches -> median weight
    of the others; then every weight capped at CAP x the median weight (user rules, 2026-10-17)."""
    names = [n for n in G if n != "rec"]
    rho = min(rho, cap / len(names))
    g0 = float(np.median(G["rec"]))
    med, w, note = {}, {}, {}
    for n in names:
        nz = [v for v in G[n] if v > 0]
        med[n] = float(np.median(nz)) if nz else 0.0
        if len(nz) >= MIN_SIGNAL:
            w[n] = rho * g0 / med[n]
    typical = float(np.median(list(w.values())))
    for n in names:
        if n not in w:
            w[n] = typical
            note[n] = f"signal in {sum(v > 0 for v in G[n])} batches < {MIN_SIGNAL}: median weight"
    limit = CAP * float(np.median(list(w.values())))
    for n in names:
        if w[n] > limit:
            note[n] = (note.get(n, "") + f" capped from {w[n]:.4g}").strip()
            w[n] = limit
    return {n: float(f"{w[n]:.4g}") for n in names}, med, g0, rho, note


def write_configs(w):
    flow = "{" + ", ".join(f"{k}: {v}" for k, v in w.items()) + "}"
    for p in CONFIGS:
        s = p.read_text(encoding="utf-8")
        s = re.sub(r"(?m)^    weights: .*$", f"    weights: {flow}   # gradient-size calibration (calibrate.py)", s)
        p.write_text(s, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--config", default=str(CONFIGS[0]))
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        mcfg = yaml.safe_load(f)["model"]
    ccfg = mcfg["constraints"]
    model, _ = warm_start("diffusion", args.stage1, mcfg)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    labels, masks, _, _ = fixed_windows(load_plans(args.zip), args.batches * args.batch, seed=1)
    G = measure(model, labels, masks, ccfg, args.batches, args.batch)
    w, med, g0, rho, note = weights_from(G)
    lines = ["# Constraint weights (gradient-size calibration, spec section 3)", "",
             f"Warm-started 16-class diffusion ({args.stage1.split('outputs/kaggle/')[-1]}), {args.batches} batches x "
             f"{args.batch} real training windows ({'GPU' if torch.cuda.is_available() else 'CPU'}), t uniform 0-{T_MAX}, "
             f"constraint losses gated by g(t) and in their training modes (G2). G = || dL / d x0_pred ||, median over "
             f"the batches with signal (G_i > 0). w_i = rho * G_0 / G_i with rho = {rho:.4f} (= min(0.05, 0.4 / {len(w)})); "
             f"signal in < {MIN_SIGNAL} batches -> median weight of the others; every weight capped at {CAP:g} x the "
             f"median weight.", "",
             f"G_0 (reconstruction): median {g0:.4g}", "",
             "| constraint | mode | median G_i | batches with signal | weight w_i | note |", "|---|---|---|---|---|---|"]
    for n in w:
        mode = ccfg.get("modes", {}).get(n, "default")
        lines.append(f"| {n} | {mode} | {med[n]:.4g} | {sum(v > 0 for v in G[n])}/{len(G[n])} | {w[n]} | {note.get(n, '')} |")
    lines += ["", "Raw G per batch (rows = constraint, values in batch order):", "", "```"]
    lines += [f"{n:4s} " + " ".join(f"{v:.3g}" for v in vals) for n, vals in G.items()]
    lines += ["```"]
    WEIGHTS_YAML.write_text(yaml.safe_dump({"weights": w, "G0": g0, "rho": rho}, sort_keys=False), encoding="utf-8")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if args.write:
        write_configs(w)
        print("weights written to", [str(p.name) for p in CONFIGS])


if __name__ == "__main__":
    main()
