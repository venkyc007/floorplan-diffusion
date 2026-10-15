"""Make the sample gallery of the interface: 24 VAL benchmark cases (8 per hole size,
cycling through the corruption types), saved to data/app/val_gallery.npz (git-ignored,
like all data). Each case keeps its benchmark mask and its true plan.

    python -m src.interface.make_gallery
"""

import numpy as np

from src.data.corruptions import TYPES
from src.eval.metrics import load_benchmark
from src.interface.inference import load_config
from src.utils import project_path

SEED = 0
SIZE_BINS = ["small", "medium", "large"]


def pick_cases(meta, per_bin, seed=SEED):
    """Indices: per size bin, `per_bin` cases cycling through the corruption types that occur
    in that bin (e.g. a single door is never a large hole); a random case within each."""
    rng = np.random.default_rng(seed)
    picks = []
    for b in SIZE_BINS:
        types = [t for t in TYPES if ((meta.size_bin == b) & (meta.type == t)).sum() >= per_bin]
        for k in range(per_bin):
            pool = meta.index[(meta.size_bin == b) & (meta.type == types[k % len(types)])].to_numpy()
            picks.append(int(rng.choice(np.setdiff1d(pool, picks))))
    return np.array(picks)


def main():
    cfg = load_config()
    bench, meta = load_benchmark(project_path("data/benchmarks/val_benchmark.npz"))
    idx = pick_cases(meta, cfg["gallery_per_size_bin"])
    out = project_path(cfg["gallery"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, inputs=bench["inputs"][idx], masks=bench["masks"][idx],
                        targets=bench["targets"][idx], plan_ids=bench["plan_ids"][idx],
                        types=meta.type.to_numpy()[idx].astype(str), size_bins=meta.size_bin.to_numpy()[idx].astype(str))
    print(f"{len(idx)} val cases -> {out}")
    print(meta.iloc[idx][["plan_id", "type", "size_bin", "mask_pct"]].to_string())


if __name__ == "__main__":
    main()
