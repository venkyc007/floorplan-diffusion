"""Rule-based missing-component detection on a FULL parsed plan (16-class map, completion scale).

    cands = detect(label, thresholds)        # -> [Candidate]

Every constraint that needs no hole is run in ABSOLUTE mode (C1 C3 C4 C6 C7 C9 C10 C11 C12
C13 C14 C15 C16 C18; C8 reachability and C17 required fixtures as exact discrete rules on the
label map), plus the check-only rules R1 (room area), R3 (forbidden door pair), R4 (no exterior
door). Each fired rule gives candidates:
    type   what is probably missing (wall, door, window, railing, room, or a fixture)
    box    (y0, x0, y1, x1) = bounding box of the evidence + 3 px padding, clipped
    rules  which rule(s) fired; score = summed violation (pixels) of the evidence
Overlapping candidates are merged (type of the highest score, rules united).
"Room with no door" (C8) is placed on the wall the room shares with the most likely reachable
neighbour (hallway > living > any other space).
A rule's candidates are kept only if score >= its threshold (tuned on intact val plans so the
whole detector raises at most ~0.5 false alarms per plan; configs/stage2/detect_rules.yaml).
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import yaml

from src.stage2.constraints.constraints import CONST, Ctx, one_hot, violation_map
from src.stage2.constraints.rules import components, r3_adjacency, room_instances
from src.stage2.data.classes16 import C16
from src.utils import PROJECT_ROOT

I = C16.index
THRESHOLDS_FILE = PROJECT_ROOT / "configs" / "stage2" / "detect_rules.yaml"
MAP_RULES = {"C1": "wall", "C3": "wall", "C4": "wall", "C6": "wall", "C7": "room", "C9": "wall", "C10": "wall",
             "C11": "door", "C12": "window", "C13": "opening", "C14": "door", "C15": "room", "C16": "fixture",
             "C18": "railing"}
DISCRETE_RULES = ["C8", "C17", "R1", "R3", "R4"]
ALL_RULES = list(MAP_RULES) + DISCRETE_RULES
PAD = 3


@dataclass
class Candidate:
    type: str
    box: tuple
    rules: list = field(default_factory=list)
    score: float = 0.0


def load_thresholds(path=THRESHOLDS_FILE):
    if not path.exists():
        return {r: 1.0 for r in ALL_RULES}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["thresholds"]


def box_of(mask, shape, pad=PAD):
    ys, xs = np.nonzero(mask)
    h, w = shape
    return (max(0, int(ys.min()) - pad), max(0, int(xs.min()) - pad), min(h - 1, int(ys.max()) + pad),
            min(w - 1, int(xs.max()) + pad))


# ---------------------------------------------------------------- map rules

def violation_maps(label, names=tuple(MAP_RULES)):
    """{rule: (H, W) violation map} for the hole-free constraints, absolute mode, on one plan."""
    with torch.no_grad():
        P = one_hot(torch.from_numpy(label.astype(np.int64))[None])
        ctx = Ctx(const=CONST, K=32)
        return {n: violation_map(n, P, ctx)[0, 0].numpy() for n in names}


def map_candidates(label, maps, level=0.5):
    out = []
    for rule, v in maps.items():
        for m in components(v > level):
            t = MAP_RULES[rule]
            if t == "opening":
                t = "door" if (label[m] == I["door"]).sum() >= (label[m] == I["window"]).sum() else "window"
            elif t == "fixture":
                t = C16.names[int(np.bincount(label[m]).argmax())]
            out.append(Candidate(t, box_of(m, label.shape), [rule], float(v[m].sum())))
    return out


# ---------------------------------------------------------------- discrete rules

def walkable_components(label):
    """Walkable components; the main one = the component holding the most ROOM pixels (the flat
    itself; the largest walkable area can be a yard or garden)."""
    walk = np.isin(label, C16.group("PASS")).astype(np.uint8)
    n, comp = cv2.connectedComponents(walk, connectivity=4)
    if n <= 1:
        return comp, 0
    rooms = np.bincount(comp[np.isin(label, C16.group("S"))], minlength=n)
    rooms[0] = -1
    return comp, int(np.argmax(rooms))


def reachable(label, comp, main):
    """Walkable pixels reachable from an entrance: the main component plus every component that
    contains an exterior door (a door within T_wall + 2 px of outdoor/background). Plans with
    several flats or floors in one image have one entrance per unit."""
    k = CONST["T_wall"] + 2
    outside = cv2.dilate(np.isin(label, C16.group("OUT")).astype(np.uint8), np.ones((2 * k + 1,) * 2, np.uint8)) > 0
    entr = set(np.unique(comp[(label == I["door"]) & outside]).tolist()) - {0}
    return np.isin(comp, list(entr | {main}))


def door_location(label, room, reach):
    """Box on the wall the room shares with a reachable neighbour (hallway > living > other)."""
    k = CONST["T_wall"] + 2
    ring = cv2.dilate(room.astype(np.uint8), np.ones((2 * k + 1,) * 2, np.uint8)) > 0
    wall = ring & (label == I["wall"])
    for group in (["hallway"], ["living"], [C16.names[c] for c in C16.group("S") + C16.group("OD")]):
        other = reach & np.isin(label, [I[g] for g in group]) & ~room
        near = cv2.dilate(other.astype(np.uint8), np.ones((2 * k + 1,) * 2, np.uint8)) > 0
        shared = wall & near
        if shared.any():
            comps = components(shared)
            return max(comps, key=lambda m: m.sum())
    return None


def discrete_candidates(label):
    out = []
    comp, main = walkable_components(label)
    reach = reachable(label, comp, main)
    # C8: room instances with no pixel in the main walkable component -> door missing
    for c, m in room_instances(label):
        if main and not (reach & m).any():
            loc = door_location(label, m, reach)
            out.append(Candidate("door", box_of(loc if loc is not None else m, label.shape), ["C8"], float(m.sum())))
    # C17: bathroom without toilet/bath, kitchen without sink/appliance (room incl. its fixtures)
    need = {"bathroom": (["toilet", "bath_shower"], "toilet"), "kitchen": (["sink", "appliance"], "sink")}
    for c, m in room_instances(label):
        name = C16.names[c]
        if name in need and not np.isin(label[m], [I[f] for f in need[name][0]]).any():
            out.append(Candidate(need[name][1], box_of(m, label.shape), ["C17"], float(m.sum())))
    # R1: room area out of range
    ranges = CONST.get("room_area", {})
    for c, m in room_instances(label):
        r = ranges.get(C16.names[c])
        if r and not (r[0] <= m.sum() <= r[1]):
            out.append(Candidate("room", box_of(m, label.shape), ["R1"], float(m.sum())))
    # R3: door joining a forbidden pair
    bad, mask = r3_adjacency(label)
    if bad:
        for m in components(mask):
            out.append(Candidate("door", box_of(m, label.shape), ["R3"], float(m.sum())))
    # R4: no exterior door -> on the exterior wall of the hallway (else living, else any room)
    from src.stage2.constraints.rules import r4_exterior_door
    if r4_exterior_door(label)[0]:
        k = CONST["T_wall"] + 2
        outside = cv2.dilate(np.isin(label, C16.group("OUT")).astype(np.uint8), np.ones((2 * k + 1,) * 2, np.uint8)) > 0
        for group in ("hallway", "living"):
            near = cv2.dilate((label == I[group]).astype(np.uint8), np.ones((2 * k + 1,) * 2, np.uint8)) > 0
            ext = (label == I["wall"]) & outside & near
            if ext.any():
                m = max(components(ext), key=lambda x: x.sum())
                out.append(Candidate("door", box_of(m, label.shape), ["R4"], float(m.sum())))
                break
    return out


# ---------------------------------------------------------------- merge + detect

def overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def merge(cands):
    """Merge candidates whose boxes overlap: union box, type of the highest score, rules united."""
    cands = sorted(cands, key=lambda c: -c.score)
    merged = []
    for c in cands:
        for m in merged:
            if overlap(m.box, c.box):
                m.box = (min(m.box[0], c.box[0]), min(m.box[1], c.box[1]), max(m.box[2], c.box[2]), max(m.box[3], c.box[3]))
                m.rules = sorted(set(m.rules) | set(c.rules))
                m.score += c.score
                break
        else:
            merged.append(Candidate(c.type, c.box, list(c.rules), c.score))
    return merged


def raw_candidates(label):
    """All candidates of every rule, before thresholds and merging."""
    return map_candidates(label, violation_maps(label)) + discrete_candidates(label)


def detect(label, thresholds=None, raw=None):
    """Thresholded + merged candidates for one full plan."""
    th = thresholds or load_thresholds()
    raw = raw if raw is not None else raw_candidates(label)
    return merge([c for c in raw if c.score >= th.get(c.rules[0], np.inf)])
