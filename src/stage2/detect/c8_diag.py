"""Why does C8 miss rooms cut off by an erased door (kind a) on parsed DRAWINGS? (user, 2026-10-19)

    python -m src.stage2.detect.c8_diag --zip parser_val.zip --heldout heldout_ids.txt [--n 15] [--extra colorful-1261]
        -> outputs/stage2/c8_diag.md, outputs/stage2/c8_diag.png

Cases: the first n held-out typed drawings (file order) that have a door whose closure (door -> wall on the
ground-truth labels) cuts a room off; that door is erased as kind (a) (wall drawn across) with erasure v2 and
the drawing goes through the pipeline's PARSE. `--extra` drawings are added the same way (multi-floor check).
Per case, on the parse: the parsed room under the cut-off room, whether C8 flagged it, and if not, why:
  door redrawn   the parser labels the erased doorway as door again;
  reached through another parsed door   the parsed room has a door elsewhere (drawn or hallucinated);
  reached through a gap in the parsed walls   no other door: a hole in the parsed walls joins it to a reached space
                 (or it seeds itself as its building part's main space);
  below C8 area threshold   C8 sees the room but its score (room area) is under the tuned threshold.
"""

import argparse
import zipfile
from pathlib import Path

import cv2
import numpy as np

from src.stage2.constraints.rules import components, room_instances
from src.stage2.data.classes16 import C16
from src.stage2.data.erasure import draw_wall, grown
from src.stage2.detect import rules as R
from src.stage2.pipeline.run import cfg, detect_parts, parse
from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "stage2"
I = C16.index


def read(z, name, flag):
    return cv2.imdecode(np.frombuffer(z.read(name), np.uint8), flag)


def cut_door(lbl):
    """(door mask, cut-off room mask) for the first door whose closure cuts a room off, or None."""
    for d in components(lbl == I["door"]):
        if d.sum() < 40:
            continue
        plan = lbl.copy()
        plan[d] = I["wall"]
        reach = R.reachable_parts(plan)
        for _, m in room_instances(plan):
            if (grown(d, 3) & m).any() and not (reach & m).any():
                return d, m
    return None


def diagnose(label, room_gt, door_gt):
    rs = lambda m: cv2.resize(m.astype(np.uint8), (label.shape[1], label.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    room, door = rs(room_gt), rs(door_gt)
    parts = detect_parts(label)
    flagged = any(grown(room | door, 3)[p.box[0]:p.box[2] + 1, p.box[1]:p.box[3] + 1].any() for p in parts.get("C8", []))
    if flagged:
        return "flagged", parts
    if door.any() and np.mean(label[door] == I["door"]) > 0.3:
        return "door redrawn", parts
    comp, _ = R.walkable_components(label)
    ids = np.bincount(comp[room & (comp > 0)].ravel()) if (room & (comp > 0)).any() else None
    if ids is None:
        return "room not parsed as walkable", parts
    cid = int(ids.argmax())
    mine = comp == cid
    reach = R.reachable_parts(label)
    if (reach & mine).any():
        pr = room_instances(label)
        here = max(pr, key=lambda t: (t[1] & room).sum())[1] if pr else room
        doors = grown(here, 3) & (label == I["door"]) & ~grown(door, 3)
        if doors.any():
            return "reached through another parsed door", parts
        return "reached through a gap in the parsed walls", parts
    raw = [c for c in R.discrete_candidates(label) if c.rules[0] == "C8"
           and grown(room | door, 3)[c.box[0]:c.box[2] + 1, c.box[1]:c.box[3] + 1].any()]
    th = R.load_thresholds().get("C8", 0)
    if raw and max(c.score for c in raw) < th:
        return f"below C8 area threshold ({int(max(c.score for c in raw))} < {int(th)} px)", parts
    return "unflagged (other)", parts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--extra", nargs="*", default=["colorful-1261"])
    args = ap.parse_args()
    keep = set(Path(args.heldout).read_text(encoding="utf-8").split())
    z = zipfile.ZipFile(args.zip)
    all_names = sorted(n[:-8] for n in z.namelist() if n.endswith("_img.png"))
    names = [n for n in all_names if n in keep and n not in args.extra]
    pc = cfg()["parse"]
    rows, figs, got = [], [], 0
    for n in names + args.extra:
        extra = n in args.extra
        if not extra and got >= args.n:
            continue
        lbl = read(z, f"{n}_lbl.png", cv2.IMREAD_UNCHANGED)
        if (lbl == 255).any() and not extra:
            continue
        c = cut_door(lbl)
        if c is None:
            if extra:
                rows.append((n, "no cut-off door in this drawing", 0))
            continue
        img = read(z, f"{n}_img.png", cv2.IMREAD_GRAYSCALE)
        d, room = c
        label, _, _ = parse(draw_wall(img, d, lbl, 8), pc)
        why, parts = diagnose(label, room, d)
        rows.append((n + (" (extra)" if extra else ""), why, len(parts.get("C8", []))))
        if len(figs) < 6 or extra:
            figs.append((n, label, room, d, parts))
        got += not extra
        print(n, why, flush=True)
    L = ["# C8 on parsed drawings: why cut-off rooms are missed", "",
         f"{got} held-out drawings with a door whose closure cuts a room off (door erased as kind (a), erasure v2, "
         "parser from configs/stage2/pipeline.yaml) + extra drawings. Categories: see src/stage2/detect/c8_diag.py.", "",
         "| drawing | outcome | C8 holes in drawing |", "|---|---|---|"]
    L += [f"| {n} | {w} | {k} |" for n, w, k in rows]
    cats = [w for _, w, _ in rows]
    L += ["", "Totals: " + ", ".join(f"{c} {cats.count(c)}" for c in sorted(set(cats))) + "."]
    (OUT / "c8_diag.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, len(figs), figsize=(6 * len(figs), 6), squeeze=False)
    for (n, label, room, d, parts), a in zip(figs, axes[0]):
        a.imshow(C16.colourise(label), interpolation="nearest")
        rs = cv2.resize(room.astype(np.uint8), (label.shape[1], label.shape[0]), interpolation=cv2.INTER_NEAREST)
        a.contour(rs, levels=[0.5], colors="orange", linewidths=1)
        for p in parts.get("C8", []):
            a.add_patch(Rectangle((p.box[1], p.box[0]), p.box[3] - p.box[1], p.box[2] - p.box[0], fill=False, ec="red"))
        a.set_title(f"{n}: cut-off room (orange), C8 holes (red)", fontsize=8)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "c8_diag.png", dpi=70)


if __name__ == "__main__":
    main()
