"""Gates G3 (overfit) and G4 (speed) for the constrained 16-class diffusion (spec section 4).

    python -m src.stage2.complete.gates34 g3 --zip completion_train.zip   -> outputs/stage2/gate_g3.md
    python -m src.stage2.complete.gates34 g4 --zip completion_train.zip   -> outputs/stage2/gate_g4.md

G3: a TINY diffusion model (as in the Stage 1 overfit test) is trained on 64 fixed real
windows for 400 steps (Adam 3e-3, batch 16, seed 0), once with constraints OFF and once ON
with the calibrated weights (ramp shortened to 50 steps so the full weights act). Then
20-step stochastic DDIM on the same 64 windows: pixel accuracy and mIoU inside the holes.
Pass: ON is at most 2 points below OFF.

G4: the FULL-size model (warm start not needed for timing): seconds per training step
(forward + backward + AdamW) with and without the constraint losses. Pass: ratio <= 1.5.
Measured on the laptop CPU (no local GPU); the K1 log gives the GPU number.
"""

import argparse
import copy
import time

import numpy as np
import torch
import yaml

from src.stage2.complete.data16 import fixed_windows, load_plans
from src.stage2.complete.models16 import Diffusion16
from src.stage2.complete.windows import model_inputs
from src.stage2.data.classes16 import C16
from src.utils import PROJECT_ROOT

CFG = PROJECT_ROOT / "configs" / "stage2" / "train16_constrained.yaml"
TINY = {"block_out_channels": [16, 32, 32, 32], "layers_per_block": 1, "norm_num_groups": 8, "attention_head_dim": 8}
OUT = PROJECT_ROOT / "outputs" / "stage2"
OUT.mkdir(parents=True, exist_ok=True)


def calibrated_weights():
    """Weights from calibrate.py (outputs/stage2/constraint_weights.yaml), else the config's."""
    p = OUT / "constraint_weights.yaml"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f)["weights"]
    return load_mcfg()["constraints"]["weights"]


def load_mcfg():
    with open(CFG, encoding="utf-8") as f:
        return yaml.safe_load(f)["model"]


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hole_scores(pred, target, mask):
    """Pixel accuracy and pooled mIoU (classes present in prediction or truth) inside the holes."""
    pred, target, mask = pred.cpu(), target.cpu(), mask.cpu()
    p, t = pred[mask[:, 0] > 0].numpy(), target.argmax(1)[mask[:, 0] > 0].numpy()
    acc = float((p == t).mean())
    ious = []
    for c in range(C16.n):
        inter, union = ((p == c) & (t == c)).sum(), ((p == c) | (t == c)).sum()
        if union:
            ious.append(inter / union)
    return acc, float(np.mean(ious))


def gradient_share(model, total, weights, step):
    """Weighted constraint loss and its share of the parameter-gradient norm (proves the
    constraint terms reach the optimizer)."""
    params = [p for p in model.parameters() if p.requires_grad]
    r = min(1.0, step / max(1, model.mcfg["constraints"].get("ramp_steps", 1)))
    cons = sum(r * weights.get(k, 0.0) * v for k, v in model.last_parts_live.items())
    g_tot = torch.autograd.grad(total, params, retain_graph=True, allow_unused=True)
    g_con = torch.autograd.grad(cons, params, retain_graph=True, allow_unused=True) if torch.is_tensor(cons) \
        and cons.requires_grad else [None] * len(params)
    norm = lambda gs: float(torch.sqrt(sum((g ** 2).sum() for g in gs if g is not None)))
    out = {"step": step, "rec": float(model.last_parts["rec"]), "constraint": float(cons),
           "share": norm(g_con) / max(norm(g_tot), 1e-12)}
    print("G3 ON", out, flush=True)
    return out


def run_g3(args):
    mcfg = load_mcfg()
    labels, masks, _, _ = fixed_windows(load_plans(args.zip), 64, seed=2)
    cond, target, mask = (v.to(DEV) for v in model_inputs(labels, masks))
    rows, share_log = {}, []
    for variant in ("off", "on"):
        torch.manual_seed(0)
        m = copy.deepcopy(mcfg) | TINY | {"ddim_steps": 20}
        weights = calibrated_weights()
        if variant == "on" and not any(weights.values()):
            raise SystemExit("G3: no calibrated weights (all 0) - the ON run would equal OFF")
        m["constraints"] = {**m["constraints"], "enabled": variant == "on", "ramp_steps": 50, "weights": weights}
        model = Diffusion16(m).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        t0 = time.time()
        g = torch.Generator().manual_seed(0)
        for step in range(args.steps):
            idx = torch.randperm(64, generator=g)[:16].to(DEV)
            loss = model.loss(cond[idx], target[idx], mask[idx], step=step)
            if variant == "on" and (step + 1) % 50 == 0:
                share_log.append(gradient_share(model, loss, weights, step + 1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (step + 1) % 100 == 0:
                print(variant, step + 1, {k: round(float(v), 5) for k, v in model.last_parts.items()}, flush=True)
        model.eval()
        pred = model.predict(cond, steps=20, generator=torch.Generator(device=DEV).manual_seed(0))
        acc, miou = hole_scores(pred, target, mask)
        rows[variant] = {"rec loss (last)": float(model.last_parts["rec"]), "hole pixel acc": acc,
                         "hole mIoU": miou, "minutes": (time.time() - t0) / 60}
        print(variant, rows[variant], flush=True)
    d_acc = 100 * (rows["on"]["hole pixel acc"] - rows["off"]["hole pixel acc"])
    d_iou = 100 * (rows["on"]["hole mIoU"] - rows["off"]["hole mIoU"])
    ok = d_acc >= -2 and d_iou >= -2
    lines = ["# Gate G3: overfit test, constraints off vs on", "",
             f"Tiny diffusion ({TINY['block_out_channels']}), 64 fixed real windows, {args.steps} steps, Adam 3e-3, "
             "batch 16, seed 0; ON = all active constraints with the calibrated weights, modes from G2, "
             "ramp 50 steps. Scores after 20-step stochastic DDIM (seed 0) inside the holes.", "",
             "| variant | rec loss (last batch) | hole pixel acc | hole mIoU | minutes |", "|---|---|---|---|---|"]
    for k, r in rows.items():
        lines.append(f"| {k} | {r['rec loss (last)']:.4f} | {100 * r['hole pixel acc']:.1f} | "
                     f"{100 * r['hole mIoU']:.1f} | {r['minutes']:.1f} |")
    if share_log:
        lines += ["", "ON run: weighted constraint loss and its share of the parameter gradient "
                  "(|| grad of sum_i w_i L_i || / || grad of the total loss ||), every 50 steps:", "",
                  "| step | rec loss | weighted constraint loss | gradient share |", "|---|---|---|---|"]
        lines += [f"| {r['step']} | {r['rec']:.4f} | {r['constraint']:.5f} | {r['share']:.3f} |" for r in share_log]
    lines += ["", f"Difference ON - OFF: accuracy {d_acc:+.1f} points, mIoU {d_iou:+.1f} points -> "
              f"**{'PASS' if ok else 'FAIL'}** (limit -2 points)."]
    (OUT / "gate_g3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def run_g4(args):
    mcfg = load_mcfg()
    labels, masks, _, _ = fixed_windows(load_plans(args.zip), args.batch * (args.steps + 1), seed=3)
    res = {}
    for variant in ("off", "on"):
        torch.manual_seed(0)
        m = copy.deepcopy(mcfg)
        m["constraints"] = {**m["constraints"], "enabled": variant == "on", "K": args.K or m["constraints"]["K"]}
        model = Diffusion16(m).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
        amp = DEV.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        times = []
        for s in range(args.steps + 1):          # the first step is a warm-up, not timed
            sl = slice(s * args.batch, (s + 1) * args.batch)
            cond, target, mask = (v.to(DEV) for v in model_inputs(labels[sl], masks[sl]))
            if DEV.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast(device_type=DEV.type, enabled=amp):
                loss = model.loss(cond, target, mask, step=10**6)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if DEV.type == "cuda":
                torch.cuda.synchronize()
            if s:
                times.append(time.time() - t0)
        res[variant] = float(np.median(times))
        print(variant, res[variant], flush=True)
    ratio = res["on"] / res["off"]
    lines = ["# Gate G4: step time with vs without constraints", "",
             f"Full-size 16-class diffusion ({mcfg['block_out_channels']}), batch {args.batch} windows of 128 x 128, "
             f"median over {args.steps} steps (forward + backward + AdamW, fp16 autocast on GPU), device: "
             f"{torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'CPU'}.", "",
             "| variant | s / step |", "|---|---|",
             f"| constraints off | {res['off']:.2f} |", f"| constraints on | {res['on']:.2f} |", "",
             f"Ratio on / off = {ratio:.2f} -> **{'PASS' if ratio <= 1.5 else 'FAIL'}** (limit 1.5). "
             f"Flood-fill steps K = {args.K or mcfg['constraints']['K']}.", "",
             f"Steps the constrained run finishes in 10 h at this speed (minus ~5% for validation): "
             f"{int(10 * 3600 / res['on'] * 0.95)}."]
    (OUT / f"gate_g4{'_K' + str(args.K) if args.K else ''}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gate", choices=["g3", "g4"])
    ap.add_argument("--zip", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--K", type=int, default=None, help="G4: flood-fill steps override")
    args = ap.parse_args()
    if args.gate == "g3":
        args.steps = args.steps or 400
        run_g3(args)
    else:
        args.steps = args.steps or 5
        run_g4(args)


if __name__ == "__main__":
    main()
