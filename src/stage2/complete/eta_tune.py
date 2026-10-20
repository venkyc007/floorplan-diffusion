"""Guidance strength (eta) on VAL holes (user, 2026-10-20): which eta gives the fewest constraint violations
without losing tolerant mIoU?

    python -m src.stage2.complete.eta_tune --zip completion_val.zip [--per-type 25] [--etas 0 0.1 0.3 1 3]
        -> outputs/stage2/eta_tune.md (+ eta_tune.json with the per-hole numbers)

Holes: 150 = 25 per type (door, window, wall, railing, fixture, room; Stage 2 training windows, typed
deletion + 2-6 px padding, src/stage2/complete/windows.py), drawn ONLY from the TUNING val plans
(configs/stage2/val_split/tuning_ids.txt), so the held-out drawings stay unseen.
Fill = the pipeline's fast hybrid: U-Net sketch + sketch prior of the hole's type (as the pipeline does) ->
re-noised to t0 = 300 -> 15 DDIM steps, guidance on the last 5, best-of-N with N = 2. The raw sampler output
is scored (the type restriction is a separate pipeline step). Each hole uses the same random seed for every
eta, so the rows differ only by eta.
Scores inside each hole (as outputs/stage2/t0_sweep.md): tolerant mIoU, validity (% of holes with no more
thresholded violations than the ground-truth window), violations per hole (total and by group).
Diagnostics: % of hole pixels that differ from the eta = 0 output, and the size of the guidance nudge
(mean |eta * gradient| on x0 inside the hole for one guided step, evaluated on the sketch as x0).
Choice (user rule): the eta with the fewest violations whose tolerant mIoU is within 1 point of eta = 0
(ties: higher tolerant mIoU, then the smaller eta), applied to the requested grid --etas only.
--extra-etas (diagnostic, not used for the choice): the gradient of the calibrated constraint losses through
softmax(x0 / 0.2) is ~1e-8 - 1e-6 per pixel on the (near one-hot) x0 of the last steps, so eta <= 3 cannot
flip a pixel; guidance starts to act around eta = 1e6 (laptop check on 3 holes, 2026-10-20). The extra rows
show what a much larger eta does, scored the same way.
"""

import argparse
import json
import time

import numpy as np
import torch

from src.stage2.complete import sampler
from src.stage2.complete.data16 import fixed_windows, load_plans
from src.stage2.complete.sketch_prior import apply_prior
from src.stage2.complete.t0_sweep import GROUPS, hole_ious
from src.stage2.complete.windows import model_inputs
from src.stage2.data.classes16 import C16
from src.stage2.pipeline.run import cfg, models
from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "stage2"
TYPES = ["door", "window", "wall", "railing", "fixture", "room"]
TUNING = PROJECT_ROOT / "configs" / "stage2" / "val_split" / "tuning_ids.txt"
T0 = 300


def holes(zip_path, per_type, limit_plans=None):
    """(labels, masks, types, prior types): per_type windows of each type from the tuning plans."""
    ids = set(TUNING.read_text(encoding="utf-8").split())
    plans = [p for p in load_plans(zip_path) if p[0] in ids][:limit_plans]
    labs, masks, kinds = [], [], []
    for k, typ in enumerate(TYPES):
        l, m, kk, _ = fixed_windows(plans, per_type, seed=100 + k, types=[typ])
        labs.append(l)
        masks.append(m)
        kinds += kk
    labs, masks = np.concatenate(labs), np.concatenate(masks)
    prior = []
    for lab, m, k in zip(labs, masks, kinds):         # the type the pipeline would paint into the sketch
        if k == "fixture":
            f = lab[(m > 0) & np.isin(lab, C16.group("F"))]
            prior.append(C16.names[int(np.bincount(f).argmax())] if len(f) else None)
        else:
            prior.append(k)
    return labs, masks, kinds, prior


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--per-type", type=int, default=25)
    ap.add_argument("--etas", type=float, nargs="+", default=[0, 0.1, 0.3, 1, 3])
    ap.add_argument("--extra-etas", type=float, nargs="*", default=[1e5, 3e5, 1e6, 3e6],
                    help="diagnostic rows (not used for the choice)")
    ap.add_argument("--limit-plans", type=int, default=None, help="smoke test: only the first N tuning plans")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fc, fast = cfg()["fill"], cfg()["fast"]
    m = models()
    unet, diff = m["unet"].to(dev), m["diffusion"].to(dev)
    labels, masks, kinds, prior = holes(args.zip, args.per_type, args.limit_plans)
    n = len(labels)
    print(f"{n} holes on {dev}:", {k: kinds.count(k) for k in TYPES}, flush=True)
    cond_all, _, mask_all = model_inputs(labels, masks)
    gt_v = sum(sampler.violation_breakdown(torch.from_numpy(labels.astype(np.int64)), mask_all,
                                           cond_all[:, :C16.n]).values())
    sketches = []
    for i in range(n):
        cond = cond_all[i:i + 1].to(dev)
        sk = unet.predict(cond)[0].cpu().numpy().astype(np.uint8)
        if fc.get("sketch_prior") and prior[i]:
            sk = apply_prior(sk, masks[i] > 0, prior[i])
        sketches.append(torch.from_numpy(sk.astype(np.int64))[None].to(dev))
    preds, secs = {}, {}
    all_etas = list(args.etas) + [e for e in args.extra_etas if e not in args.etas]
    for eta in all_etas:
        t, out = time.time(), []
        for i in range(n):
            g = torch.Generator(device=dev).manual_seed(1000 + i)          # same seed for every eta
            res, _ = sampler.fill(diff, cond_all[i:i + 1].to(dev), sketch=sketches[i], n=fast["n_samples"], t0=T0,
                                  eta_guidance=eta, generator=g, steps_from_t0=fast["steps_from_t0"],
                                  guide_last=fast["guide_last"])
            out.append(res[0].cpu().numpy().astype(np.uint8))
        preds[eta], secs[eta] = np.stack(out), (time.time() - t) / n
        print(f"eta {eta}: {secs[eta]:.2f} s / hole", flush=True)
    grad = []                                           # guidance gradient size per hole (eta = 1)
    w = sampler.guidance_weights()
    for i in range(n):
        x0 = (torch.nn.functional.one_hot(sketches[i], C16.n).permute(0, 3, 1, 2).float() * 2 - 1)
        c = cond_all[i:i + 1].to(dev)
        d = sampler.guide_x0(x0, c[:, -1:], c[:, :C16.n], w, 1.0) - x0
        grad.append(float(d.abs().sum() / (c[:, -1:].sum() * C16.n)))
    grad = float(np.mean(grad))
    p0 = preds[args.etas[0]]
    rows = []
    for eta in all_etas:
        p = preds[eta]
        diff = 100 * np.mean([(p[i] != p0[i])[masks[i] > 0].mean() for i in range(n)])
        ious = np.array([hole_ious(p[i], labels[i], masks[i] > 0) for i in range(n)])
        bd = sampler.violation_breakdown(torch.from_numpy(p.astype(np.int64)), mask_all, cond_all[:, :C16.n])
        tot = sum(bd.values())
        rows.append({"eta": eta, "tol": 100 * ious[:, 1].mean(), "strict": 100 * ious[:, 0].mean(),
                     "valid": 100 * np.mean(tot <= gt_v), "viol": float(tot.mean()),
                     "groups": {g: float(np.mean(sum(bd[k] for k in GROUPS[g]))) for g in GROUPS},
                     "per_type": {k: {"tol": 100 * float(ious[[j for j in range(n) if kinds[j] == k], 1].mean()),
                                      "viol": float(tot[[j for j in range(n) if kinds[j] == k]].mean())}
                                  for k in TYPES if k in kinds},
                     "s_per_hole": secs[eta], "changed_pct": float(diff), "nudge": eta * grad, "per_hole_viol": tot.tolist(), "per_hole_tol": ious[:, 1].tolist()})
    base = next(r for r in rows if r["eta"] == 0) if 0 in args.etas else rows[0]
    rule = lambda rs: min([r for r in rs if r["tol"] >= base["tol"] - 1.0], key=lambda r: (r["viol"], -r["tol"], r["eta"]))
    best = rule([r for r in rows if r["eta"] in args.etas])
    best_all = rule(rows)
    L = [f"# Guidance strength (eta) on val holes{args.tag}", "",
         f"{n} holes from the tuning val plans ({', '.join(f'{k} {kinds.count(k)}' for k in TYPES)}); fast hybrid "
         f"(U-Net sketch + sketch prior, t0 = {T0}, {fast['steps_from_t0']} DDIM steps, guidance on the last "
         f"{fast['guide_last']}, N = {fast['n_samples']}); same seed per hole for every eta. Ground-truth windows: "
         f"{gt_v.mean():.2f} violations per hole. Means per hole.", "",
         "| eta | tolerant mIoU | strict mIoU | validity % | violations / hole | walls | openings | spaces | fixtures | "
         f"hole px changed vs eta = {args.etas[0]:g} | x0 nudge / guided step | s / hole |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['eta']:g}{' (chosen)' if r is best else ''}{' (extra)' if r['eta'] not in args.etas else ''} | {r['tol']:.1f} | {r['strict']:.1f} | "
                 f"{r['valid']:.0f} | {r['viol']:.2f} | " + " | ".join(f"{v:.2f}" for v in r["groups"].values()) +
                 f" | {r['changed_pct']:.1f}% | {r['nudge']:.4f} | {r['s_per_hole']:.2f} |")
    L += ["", "Per type (tolerant mIoU / violations per hole):", "",
          "| eta | " + " | ".join(k for k in TYPES if k in kinds) + " |", "|---|" + "---|" * len(set(kinds))]
    for r in rows:
        L.append(f"| {r['eta']:g} | " + " | ".join(f"{v['tol']:.1f} / {v['viol']:.2f}" for v in r["per_type"].values()) + " |")
    L += ["", f"**Choice (fewest violations with tolerant mIoU within 1 point of eta = {base['eta']:g}, "
              f"{base['tol']:.1f}): eta = {best['eta']:g}** ({best['tol']:.1f} mIoU, {best['viol']:.2f} violations, "
              f"validity {best['valid']:.0f}%)."]
    if args.extra_etas:
        L += ["", f"Same rule over all rows incl. the extra (diagnostic) etas: eta = {best_all['eta']:g} "
                  f"({best_all['tol']:.1f} mIoU, {best_all['viol']:.2f} violations, validity {best_all['valid']:.0f}%)."]
    (OUT / f"eta_tune{args.tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / f"eta_tune{args.tag}.json").write_text(json.dumps({"chosen": best["eta"], "chosen_incl_extra": best_all["eta"], "rows": rows, "kinds": kinds},
                                                             indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
