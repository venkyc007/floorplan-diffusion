"""Fixed evaluation benchmarks: the same corruptions every time.

For every plan in a split, one corruption of each of the 5 types
(door, window, wall, room, region), drawn with a fixed seed, so every model
is scored on exactly the same task. Plans with nothing valid of a type
(e.g. no windows) are skipped for that type and counted.

Run from the project folder:
    python -m src.data.make_benchmark                 # test + val
    python -m src.data.make_benchmark --splits val    # only val (e.g. on Kaggle)
    python -m src.data.make_benchmark --limit 50      # quick test

Each run prints a checksum per benchmark; full (non --limit) runs also save
it to configs/benchmark_checksums.json, so a copy rebuilt elsewhere (Kaggle)
can be checked to be byte-identical.

Saves to data/benchmarks/:
  <split>_benchmark.npz
      inputs   (N, 128, 128) uint8  corrupted label maps: masked pixels = 255
                                    (turn into the 7-channel model input with
                                    corruptions.to_model_input(inputs[i], masks[i]))
      targets  (N, 128, 128) uint8  clean label maps
      masks    (N, 128, 128) bool   region to fill in
      plan_ids (N,)
  <split>_benchmark.csv   one row per sample: plan_id, type, location,
                          position, size_bin, mask_area, ...
"""

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from src.data.corruptions import corrupt
from src.data.dataset import load_labels
from src.utils import load_config, project_path

OUT_DIR = "data/benchmarks"


def make_benchmark(labels, plan_ids, cfg, limit=None):
    """Build one benchmark. Returns (arrays dict, metadata DataFrame, skip counts)."""
    b = cfg["benchmark"]
    inputs, targets, masks, rows = [], [], [], []
    skips = {t: 0 for t in b["types"]}
    n = len(labels) if limit is None else min(limit, len(labels))
    for i in range(n):
        for k, ctype in enumerate(b["types"]):
            # Seed from (benchmark seed, plan id, type): each sample is
            # reproducible on its own, whatever order plans are processed in.
            rng = np.random.default_rng([b["seed"], int(plan_ids[i]), k])
            result = corrupt(labels[i], ctype, rng, cfg)
            if result is None:
                skips[ctype] += 1
                continue
            mask, corrupted, meta = result
            inputs.append(corrupted)
            targets.append(labels[i])
            masks.append(mask)
            rows.append({"plan_id": int(plan_ids[i]), **meta})
    arrays = {"inputs": np.stack(inputs), "targets": np.stack(targets),
              "masks": np.stack(masks), "plan_ids": np.array([r["plan_id"] for r in rows])}
    return arrays, pd.DataFrame(rows), skips


def checksum(arrays):
    """SHA-256 over the benchmark arrays (fixed key order): identical files
    give identical checksums on any machine."""
    h = hashlib.sha256()
    for key in ("inputs", "targets", "masks", "plan_ids"):
        a = np.ascontiguousarray(arrays[key])
        h.update(key.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def print_counts(meta, skips, name):
    print(f"\n{name}: {len(meta)} samples; skipped (no valid part): {skips}")
    for col in ("type", "location", "position", "size_bin"):
        print(f"  {col:9s}", meta[col].value_counts().to_dict())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only the first N plans per split")
    parser.add_argument("--splits", nargs="+", default=["test", "val"], choices=["test", "val"])
    args = parser.parse_args()
    cfg = load_config("corruption")
    proc = project_path(load_config("data")["paths"]["processed_dir"])
    out = project_path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    sums_path = project_path("configs/benchmark_checksums.json")
    sums = json.loads(sums_path.read_text(encoding="utf-8")) if sums_path.exists() else {}
    for split in args.splits:
        npz = proc / f"{split}.npz"
        arrays, meta, skips = make_benchmark(load_labels(npz), np.load(npz)["plan_ids"], cfg, args.limit)
        np.savez_compressed(out / f"{split}_benchmark.npz", **arrays)
        meta.to_csv(out / f"{split}_benchmark.csv", index_label="sample")
        print_counts(meta, skips, f"{split} benchmark")
        digest = checksum(arrays)
        print(f"  checksum {digest}")
        if args.limit is None:
            sums[split] = digest
    if args.limit is None:
        sums_path.write_text(json.dumps(sums, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
