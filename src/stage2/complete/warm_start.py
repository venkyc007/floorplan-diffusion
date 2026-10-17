"""Warm start of the 16-class FILL models from the Stage 1 best checkpoints (class expansion, spec section 2).

    python -m src.stage2.complete.warm_start --kind diffusion|unet --stage1 <best.pt> --config <yaml> --out <pt>

Stage 1 classes: wall, door, window, room, outdoor, background. Every new class takes the
weights of one old class:
    wall, door, window, outdoor, background -> the same old class
    railing                                 -> wall
    6 room types + 4 fixtures               -> room
applied to the input conv (diffusion: noisy-target half AND condition half; U-Net:
condition) and to the output conv. All other weights are copied unchanged.

Result: for the condition (one-hot, 0/1) the new network computes exactly what Stage 1
computed; its outputs for the 10 room-type/fixture channels are identical (undecided).
The diffusion's noisy-target channels live in [-1, 1]: inactive classes are -1, so a
group of n copies adds a constant -(n-1)*w per filter tap. That offset is moved into the
conv_in bias, which is exact for clean targets (t = 0) and an approximation at high noise
(the noise of n channels does not cancel); measured by `check_equivalence`.
"""

import argparse

import torch
import yaml

from src.stage2.complete.models16 import build16
from src.stage2.data.classes16 import C16

OLD = ["wall", "door", "window", "room", "outdoor", "background"]
OLD_OF_NEW = {"wall": "wall", "door": "door", "window": "window", "railing": "wall",
              "outdoor": "outdoor", "background": "background"}
for _c in C16.cfg["groups"]["S"] + C16.cfg["groups"]["F"]:
    OLD_OF_NEW[_c] = "room"
MAP = [OLD.index(OLD_OF_NEW[n]) for n in C16.names]          # new channel -> old channel
N_OLD, N_NEW = len(OLD), C16.n


def expand_state(old_sd, new_model, kind):
    """Stage 1 state dict -> state dict for the 16-class model."""
    new_sd = new_model.state_dict()
    out = {}
    for k, v in old_sd.items():
        if k not in new_sd:
            raise KeyError(f"unexpected key {k}")
        if k == "net.conv_in.weight":
            parts = [v[:, :N_OLD][:, MAP], v[:, N_OLD:2 * N_OLD][:, MAP], v[:, 2 * N_OLD:]] if kind == "diffusion" \
                else [v[:, :N_OLD][:, MAP], v[:, N_OLD:]]
            out[k] = torch.cat(parts, 1)
        elif k == "net.conv_in.bias" and kind == "diffusion":
            w = old_sd["net.conv_in.weight"][:, :N_OLD]           # noisy-target half
            sizes = torch.bincount(torch.tensor(MAP), minlength=N_OLD).to(w)
            out[k] = v + ((sizes - 1)[None, :, None, None] * w).sum((1, 2, 3))
        elif k == "net.conv_out.weight":
            out[k] = v[MAP]
        elif k == "net.conv_out.bias":
            out[k] = v[MAP]
        else:
            if v.shape != new_sd[k].shape:
                raise ValueError(f"shape mismatch {k}: {tuple(v.shape)} vs {tuple(new_sd[k].shape)}")
            out[k] = v
    missing = set(new_sd) - set(out)
    if missing:
        raise KeyError(f"not initialised: {sorted(missing)[:5]}")
    return out


def warm_start(kind, stage1_ckpt, mcfg):
    model = build16({**mcfg, "kind": kind})
    ck = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(expand_state(ck["model"], model, kind))
    return model, ck.get("step")


def merge(x16):
    """16-class output -> 6 Stage 1 channels (room = mean of its copies, which are identical)."""
    return torch.stack([x16[:, [i for i, m in enumerate(MAP) if m == o]].mean(1) for o in range(N_OLD)], 1)


@torch.no_grad()
def check_equivalence(model16, model6, kind, n=4, size=64, seed=0):
    """Max |difference| between the merged 16-class output and Stage 1's on random inputs
    built from the same label maps (diffusion: at t = 0, 200, 500, 900)."""
    g = torch.Generator().manual_seed(seed)
    old_lab = torch.randint(0, N_OLD, (n, size, size), generator=g)
    room_new = [i for i, m in enumerate(MAP) if m == 3]
    new_lab = torch.tensor([MAP.index(o) for o in range(N_OLD)])[old_lab]   # first new class of each old one
    rooms = old_lab == 3
    new_lab[rooms] = torch.tensor(room_new)[torch.randint(0, len(room_new), (int(rooms.sum()),), generator=g)]
    mask = torch.zeros(n, 1, size, size)
    mask[:, :, 16:48, 16:48] = 1
    y6 = torch.nn.functional.one_hot(old_lab, N_OLD).permute(0, 3, 1, 2).float()
    y16 = torch.nn.functional.one_hot(new_lab, N_NEW).permute(0, 3, 1, 2).float()
    c6, c16 = torch.cat([y6 * (1 - mask), mask], 1), torch.cat([y16 * (1 - mask), mask], 1)
    if kind == "unet":
        return {"max_abs_diff": float((merge(model16(c16)) - model6(c6)).abs().max())}
    res = {}
    for t in (0, 200, 500, 900):
        tt = torch.full((n,), t)
        eps = torch.randn(n, N_NEW, size, size, generator=g)
        x16 = model16.train_scheduler.add_noise(y16 * 2 - 1, eps, tt)
        x6 = model6.train_scheduler.add_noise(y6 * 2 - 1, merge(eps), tt)
        out16, out6 = merge(model16(x16, c16, tt)), model6(x6, c6, tt)
        res[f"t{t}_max_abs_diff"] = float((out16 - out6).abs().max())
        res[f"t{t}_argmax_agree"] = float((out16.argmax(1) == out6.argmax(1)).float().mean())
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["diffusion", "unet"], required=True)
    ap.add_argument("--stage1", required=True, help="Stage 1 best.pt")
    ap.add_argument("--config", required=True, help="16-class training config (model section)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        mcfg = yaml.safe_load(f)["model"]
    model, step = warm_start(args.kind, args.stage1, mcfg)
    torch.save({"model": model.state_dict(), "step": 0, "source": args.stage1, "source_step": step}, args.out)
    print(f"saved {args.out} (from {args.stage1}, step {step})")


if __name__ == "__main__":
    main()
