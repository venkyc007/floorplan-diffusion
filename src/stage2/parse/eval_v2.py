"""Parser v1 vs v2A vs v2B on the HELD-OUT val drawings, normal and text-erased -> outputs/stage2/parser_v2.md.

    python -m src.stage2.parse.eval_v2 --val <folder with val drawings> --heldout heldout_ids.txt \
        --v1 v1/best.pt --v2a v2A/best.pt --v2b v2B/best.pt [--limit N]

Conditions: (a) the drawing as is, (b) text erased with src/stage2/parse/text_erase.py (ground-truth labels).
v2A always erases text first at inference (two passes, train_v2.parse_v2); v1 and v2B parse once.
Per model and condition: mIoU (16 classes), wall / door / window / railing IoU, door / window F1, fixture F1,
room-type IoU (6 types) and their mean, and the room-type confusion matrix (% of each true type's pixels).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from src.stage2.data.classes16 import C16
from src.stage2.parse.dataset import list_plans, read_pair
from src.stage2.parse.metrics import ParserMetrics
from src.stage2.parse.text_dependence import confusion, table
from src.stage2.parse.text_erase import erase_text
from src.stage2.parse.train import Parser
from src.stage2.parse.train_v2 import parse_v2
from src.utils import PROJECT_ROOT

ROOMS = C16.group("S")
OUT = PROJECT_ROOT / "outputs" / "stage2" / "parser_v2.md"
CFG = PROJECT_ROOT / "configs" / "stage2" / "parser_v2.yaml"


def load(path, mcfg, device):
    m = Parser(mcfg, encoder_weights=None).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    return m.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--v1", required=True)
    ap.add_argument("--v2a", required=True)
    ap.add_argument("--v2b", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    keep = set(Path(args.heldout).read_text(encoding="utf-8").split())
    plans = [p for p in list_plans(args.val) if p in keep][: args.limit]
    models = {"v1": (args.v1, False), "v2A (no text)": (args.v2a, True), "v2B (mixed)": (args.v2b, False)}
    vc = cfg["val"]
    met = {(m, c): ParserMetrics(vc["component_classes"], vc["component_iou"], 255)
           for m in models for c in ("normal", "text erased")}
    conf = {k: np.zeros((len(ROOMS), len(ROOMS) + 1), np.int64) for k in met}
    pc = {k: np.zeros(len(ROOMS), np.int64) for k in met}
    nets = {m: load(p, cfg["model"], device) for m, (p, _) in models.items()}
    for i, p in enumerate(plans):
        img, lbl = read_pair(args.val, p)
        imgs = {"normal": img, "text erased": erase_text(img, lbl)}
        for m, (_, no_text) in models.items():
            for c, im in imgs.items():
                with torch.no_grad():
                    pred = parse_v2(nets[m], im, device, amp, no_text)
                met[(m, c)].update(pred, lbl)
                if not (lbl == 255).any():
                    conf[(m, c)] += confusion(pred, lbl)
                    pc[(m, c)] += np.array([(pred == r).sum() for r in ROOMS])
        print(f"{i + 1}/{len(plans)} {p}", flush=True)
    res = {k: v.result() for k, v in met.items()}
    names = [C16.names[r] for r in ROOMS]
    cols = ["mIoU", "iou_wall", "iou_door", "iou_window", "iou_railing", "F1_door", "F1_window", "F1_toilet",
            "F1_sink", "F1_bath_shower", "F1_appliance"] + [f"iou_{n}" for n in names]
    L = ["# Parser v2 vs v1 (held-out val drawings)", "",
         f"{len(plans)} held-out val drawings (configs/stage2/val_split/heldout_ids.txt; none used for any choice). "
         "Text erased = src/stage2/parse/text_erase.py with the ground-truth labels. v2A erases text itself before "
         "parsing (two passes); v1 and v2B parse once. Numbers in %.", "",
         "| model | input | " + " | ".join(c.replace("iou_", "IoU ").replace("F1_", "F1 ") for c in cols) +
         " | **room-type mIoU** |", "|---|---|" + "---|" * (len(cols) + 1)]
    for (m, c), r in res.items():
        room = np.mean([r[f"iou_{n}"] for n in names])
        L.append(f"| {m} | {c} | " + " | ".join(f"{100 * r[k]:.1f}" for k in cols) + f" | **{100 * room:.1f}** |")
    for k in met:
        L += ["", f"## Room-type confusion: {k[0]}, {k[1]} (% of each true type's pixels)", ""] + table(conf[k])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
