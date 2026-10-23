"""Second Stage 2 TEST evaluation (user, 2026-10-22): END TO END on COMPONENT-LEVEL omissions, all test drawings.

    python -m src.stage2.test_components --parser-zip parser_test.zip --out DIR [--part k --parts n] [--smoke]
    python -m src.stage2.test_components --report --out DIR            # merge the shards -> DIR/test_components.md

Why: the one-shot test (src/stage2/final_test.py -> outputs/stage2/final_test.md) used a size-balanced omission mix
in which two thirds of the cases were a whole room or a region, which the pipeline cannot detect as such; the user
asked for a larger test restricted to single components. The pipeline and every setting are unchanged (frozen
configs/stage2/pipeline.yaml, detector_threshold.yaml): this is a second evaluation of the same system on the same
test split, designed after seeing the first one (docs/decisions.md, 2026-10-22).

Cases: every typed test drawing (final_test.drawing_ids), in a fixed random order (seed 2027), one omission each.
The kind rotates through image_eval.KINDS (door (a), door (b), window, wall, railing, fixture, room = a room's walls
and doors): case i tries KINDS[i % 7] first, then the next kinds if the drawing has none -> about equal counts per
kind. Erasure: image_eval.omit, the generator every validation table used (rng seed 2027 + i), with its metadata
(type, location, position, size bin). Pipeline: run_pipeline with the validated fast settings.
Scores (e2e_val.score, the validation's definitions): found = a flag overlaps the omission; right type = such a
flag has the omitted type; fixed = found AND the final map has the right class there (IoU >= 0.3), whoever drew it
(fill or parser); fixed by the fill = the fill's own additions reach IoU >= 0.3; present = the right class is in the
final map there, flagged or not; false additions = added / marked components away from the omission; tolerant
mIoU and validity in the omission region before -> after. Proportions with Wilson 95% intervals.
Each finished case is appended to DIR/part<k>.jsonl (resume: finished cases are skipped; a failing case is
recorded as an error row and the shard goes on).
"""

import argparse
import glob
import json
import os
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

from src.utils import PROJECT_ROOT

SEED = 2027


def order(parser_zip):
    from src.stage2.final_test import drawing_ids
    ids = drawing_ids(parser_zip)
    return [ids[i] for i in np.random.default_rng(SEED).permutation(len(ids))]


def do_case(z, pid, i):
    from src.stage2.detect.image_eval import KINDS, omit
    from src.stage2.final_test import read
    from src.stage2.pipeline import e2e_val
    from src.stage2.pipeline.run import run_pipeline
    img = read(z, f"{pid}_img.png", cv2.IMREAD_GRAYSCALE)
    lbl = read(z, f"{pid}_lbl.png", cv2.IMREAD_UNCHANGED)
    rng = np.random.default_rng(SEED + i)
    k0 = i % len(KINDS)
    o, kind = None, None
    for kind in KINDS[k0:] + KINDS[:k0]:
        o = omit(kind, img, lbl, rng)
        if o is not None:
            break
    if o is None:
        return None
    erased, removed, typ, _, meta, _ = o
    t = time.time()
    res = run_pipeline(erased, fast=True)
    sec = time.time() - t
    s = e2e_val.score(res, removed, lbl, typ)
    keep = ("found", "right_type", "fixed", "fix_iou", "fixed_add", "fix_iou_add", "false_adds", "tol_before",
            "tol_after", "valid_before", "valid_after", "flags")
    return {"pid": pid, "kind": kind, "type": typ, **{f"meta_{k}": v for k, v in meta.items()},
            **{k: (bool(s[k]) if isinstance(s[k], (bool, np.bool_)) else s[k]) for k in keep},
            "present": bool(s["fix_iou"] >= 0.3), "messages": res["messages"], "seconds": round(sec, 1)}


def read_rows(out):
    rows = []
    for f in sorted(glob.glob(str(Path(out) / "part*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:                  # a line cut off by a time-out: that case runs again
                continue
    return rows


def run_shard(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = {r["idx"] for r in read_rows(out) if "error" not in r}
    log = open(out / f"part{args.part}.jsonl", "a", encoding="utf-8")
    z = zipfile.ZipFile(args.parser_zip)
    ids = order(args.parser_zip)
    if args.smoke:
        ids = ids[:3]
    for i, pid in enumerate(ids):
        if i % args.parts != args.part or i in done:
            continue
        try:
            row = do_case(z, pid, i)
            row = {"pid": pid, "skipped": True} if row is None else row
        except Exception as e:                            # one odd drawing must not kill the shard
            import traceback
            traceback.print_exc()
            row = {"pid": pid, "error": repr(e)[:300]}
        log.write(json.dumps({"idx": i, **row}, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
        log.flush()
        print(f"[part {args.part}] case {i + 1}/{len(ids)}", flush=True)
    log.close()


def wilson(k, n, z=1.96):
    """95% Wilson interval of a proportion k / n, in %."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def pct(rows, key):
    k, n = sum(bool(r[key]) for r in rows), len(rows)
    lo, hi = wilson(k, n)
    return f"{100 * k / max(n, 1):.0f}% [{lo:.0f}, {hi:.0f}]" if n else "-"


def summary_row(name, R):
    if not R:
        return f"| {name} | 0 | | | | | | | | |"
    return (f"| {name} | {len(R)} | {pct(R, 'found')} | {pct(R, 'right_type')} | {pct(R, 'fixed')} | "
            f"{pct(R, 'fixed_add')} | {pct(R, 'present')} | {np.mean([r['false_adds'] for r in R]):.2f} | "
            f"{np.mean([r['tol_before'] for r in R]):.1f} -> {np.mean([r['tol_after'] for r in R]):.1f} | "
            f"{np.mean([r['seconds'] for r in R]):.0f} |")


def report(out):
    from src.stage2.data.omission_meta import FIELDS, breakdowns
    all_rows = read_rows(out)
    rows = list({r["idx"]: r for r in all_rows if "error" not in r}.values())
    errors = {r["idx"] for r in all_rows if "error" in r} - {r["idx"] for r in rows}
    cases = [r for r in rows if not r.get("skipped")]
    version = PROJECT_ROOT / "code_version.txt"
    head = ("| cases | n | found | right type | fixed (any source) | fixed by the fill | right class present (flag or not) | "
            "false additions per drawing | tol. mIoU in the region | s per drawing |")
    L = ["# Stage 2 TEST, second evaluation: end to end on component-level omissions (all test drawings)", "",
         f"Code version: {version.read_text(encoding='utf-8').strip() if version.exists() else 'unknown'}. Frozen "
         "settings; fast sampling (as validated). One single-component omission per test drawing, kinds rotated "
         "(src/stage2/test_components.py). Brackets: Wilson 95% intervals.", "",
         f"Drawings: {len(rows) + len(errors)} processed, {len(cases)} scored, "
         f"{sum(1 for r in rows if r.get('skipped'))} skipped (no component of any kind), {len(errors)} errors.", "",
         head, "|---|---|---|---|---|---|---|---|---|---|",
         summary_row("all", cases),
         summary_row("all except railings (manual-only by design)", [r for r in cases if r["kind"] != "railing"])]
    L += ["", "## By omission kind", "", head, "|---|---|---|---|---|---|---|---|---|---|"]
    for kind in ("door (a)", "door (b)", "window", "wall", "railing", "fixture", "room"):
        L.append(summary_row(kind, [r for r in cases if r["kind"] == kind]))
    mrows = [{**{f: r.get(f"meta_{f}", "") for f in FIELDS}, "r": r} for r in cases]
    p = lambda rs, k: f"{100 * np.mean([bool(x['r'][k]) for x in rs]):.0f}%"
    L += breakdowns(mrows, {"found": lambda rs: p(rs, "found"), "right type": lambda rs: p(rs, "right_type"),
                            "fixed": lambda rs: p(rs, "fixed"), "fixed by the fill": lambda rs: p(rs, "fixed_add"),
                            "false adds": lambda rs: f"{np.mean([x['r']['false_adds'] for x in rs]):.2f}"})
    (Path(out) / "test_components.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parser-zip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="first 3 drawings only (use the VAL zip)")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report(args.out)
    else:
        run_shard(args)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
