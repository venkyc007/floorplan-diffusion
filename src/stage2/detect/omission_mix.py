"""Size-balanced omissions for the final test evaluation (user, 2026-10-20): roughly a third each of small /
medium / large omissions (Stage 1 size bins: < 2.7 / < 7.7 / larger, % of the building).

    bins = plan_bins(n)                                   # ["small", "medium", "large", "small", ...]
    o = omit_in_bin("medium", img, lbl, rng)             # same tuple as image_eval.omit, or None

Settings: configs/stage2/test_eval.yaml (size_mix). Per bin:
  small   the small kinds of image_eval.KINDS (door (a), door (b), window, wall, railing, fixture, room walls),
          starting at a random one and taking the next if the drawing lacks it; kept only if it lands in "small"
  medium  room (whole) with the room at 2.7-7.7% of the building, or a region of 5-7.7% (random order)
  large   room (whole) at 7.7-15%, or a region of 7.7-25%
Every case is checked against its own metadata (omission_meta), so the bin is the measured one.
A drawing where no kind fits its bin gets the next bin in turn (test_eval.yaml: fallback), so the mix stays near a
third each; the report gives the realised counts per bin.
"""

import numpy as np
import yaml

from src.stage2.data.erasure_large import erase_region, erase_room
from src.stage2.data.omission_meta import SIZE_EDGES, omission_meta
from src.stage2.detect.image_eval import KINDS, omit
from src.utils import PROJECT_ROOT

CFG = yaml.safe_load((PROJECT_ROOT / "configs" / "stage2" / "test_eval.yaml").read_text(encoding="utf-8"))
BINS = ("small", "medium", "large")


def plan_bins(n, mix=None):
    """n target bins in a fixed cycle following the mix (default 1/3 each)."""
    mix = mix or CFG["size_mix"]
    counts = {b: int(round(n * mix[b])) for b in BINS}
    while sum(counts.values()) > n:
        counts[max(counts, key=counts.get)] -= 1
    while sum(counts.values()) < n:
        counts[min(counts, key=counts.get)] += 1
    out, left = [], dict(counts)
    while len(out) < n:
        for b in BINS:
            if left[b]:
                out.append(b)
                left[b] -= 1
    return out


def _large(kind, img, lbl, rng, room_pct, region_pct):
    o = erase_room(img, lbl, rng, pct=room_pct) if kind == "room (whole)" else erase_region(img, lbl, rng, pct=region_pct)
    if o is None:
        return None
    out, structure, area = o
    return out, structure, "room", True, omission_meta(lbl, area, kind), area


def omit_in_bin(target, img, lbl, rng):
    lo, hi = SIZE_EDGES
    if target == "small":
        k0 = int(rng.integers(len(KINDS)))
        for kind in KINDS[k0:] + KINDS[:k0]:
            o = omit(kind, img, lbl, rng)
            if o is not None and o[4]["size_bin"] == "small":
                return o
        return None
    room_pct, region_pct = ((lo, hi), (max(lo, 5.0), hi)) if target == "medium" else ((hi, 15.0), (hi, 25.0))
    kinds = ["room (whole)", "region"]
    for kind in (kinds if rng.random() < 0.5 else kinds[::-1]):
        o = _large(kind, img, lbl, rng, room_pct, region_pct)
        if o is not None and o[4]["size_bin"] == target:
            return o
    return None


def omit_balanced(target, img, lbl, rng):
    """omit_in_bin, falling back to the next bins in turn (CFG fallback) -> (bin used, omit tuple) or None."""
    order = [target] + [b for b in BINS[BINS.index(target) + 1:] + BINS[:BINS.index(target)]] if CFG["fallback"] else [target]
    for b in order:
        o = omit_in_bin(b, img, lbl, rng)
        if o is not None:
            return b, o
    return None
