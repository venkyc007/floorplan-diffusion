"""Score a trained checkpoint on a fixed benchmark (all Day 2 metrics).

    python -m src.stage1.evaluate --checkpoint outputs/runs/diffusion/checkpoints/best.pt --benchmark val
    python -m src.stage1.evaluate --checkpoint ... --benchmark test --steps 50 --limit 500

Use the VAL benchmark to compare and choose models; run TEST only for the
final numbers. Writes <checkpoint folder>/../eval_<benchmark>.csv and .md
(overall + breakdowns by type, location, position, size bin).
"""

import argparse
from pathlib import Path

import torch

from src.stage1.eval.baselines import SHOW
from src.stage1.eval.metrics import evaluate_benchmark, load_benchmark, to_markdown
from src.stage1.models import build_model
from src.utils import project_path


def load_trained(path, device):
    """Rebuild the model from a checkpoint (best.pt or last.pt)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"]["model"]).to(device)
    # last.pt holds raw + EMA weights: evaluate the EMA ones if present.
    model.load_state_dict(ckpt.get("ema") or ckpt["model"])
    return model.eval(), ckpt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark", choices=["val", "test"], default="val")
    parser.add_argument("--steps", type=int, help="DDIM steps (diffusion; default from config)")
    parser.add_argument("--limit", type=int, help="only the first N cases (quick check)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_trained(args.checkpoint, device)
    bench, meta = load_benchmark(project_path(f"data/benchmarks/{args.benchmark}_benchmark.npz"))
    if args.limit:
        bench, meta = {k: v[:args.limit] for k, v in bench.items()}, meta.iloc[:args.limit]
    gen = torch.Generator(device=device).manual_seed(0)

    def predict(batch):
        cond = torch.from_numpy(batch["inputs"]).to(device)
        with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
            return model.predict(cond, steps=args.steps, generator=gen).cpu().numpy()

    table, _ = evaluate_benchmark(predict, bench, meta, batch_size=args.batch_size)
    out = Path(args.checkpoint).parent.parent / f"eval_{args.benchmark}"
    table.to_csv(f"{out}.csv")
    title = (f"# {ckpt['config']['model']['kind']} (step {ckpt['step']}) on the {args.benchmark} "
             f"benchmark, {len(meta)} cases")
    Path(f"{out}.md").write_text(title + "\n\n" + to_markdown(table[SHOW]) + "\n", encoding="utf-8")
    print(to_markdown(table.loc[["overall"], SHOW]))
    print(f"Saved {out}.csv / .md")


if __name__ == "__main__":
    main()
