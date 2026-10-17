"""Hand-made toy plans for gate G1: one valid plan, one broken variant per constraint.

The valid plan (96 x 96, walls 3 px) satisfies every rule:

    living room across the top (window in the top wall, front door to the outside,
    door to a railed balcony on the right); below it bedroom | kitchen | bathroom, each
    entered from the living room by a door; bedroom window on the left, bathroom window
    at the bottom; toilet + bath in the bathroom, sink + appliance in the kitchen, all
    standing against walls.

TOY_CONST holds small toy-sized constants (the real ones are for whole CubiCasa plans).
Each broken toy is (label map, hole or None) and breaks the named rule.
"""

import numpy as np

from src.stage2.data.classes16 import C16

I = C16.index
SIZE = 96

TOY_CONST = {
    "T_wall": 3, "T_wall_max": 5,
    "Lmin_door": 8, "Lmax_door": 14, "Lmin_win": 8, "Lmax_win": 18,
    "k": {"living": 9, "bedroom": 9, "kitchen": 9, "bathroom": 5, "hallway": 5, "other_room": 5,
          "outdoor": 5, "toilet": 3, "sink": 3, "bath_shower": 3, "appliance": 3},
    "k_clear": 3, "k_fix": 15,
    "c6_excluded_pairs": [["living", "kitchen"], ["living", "hallway"], ["outdoor", "background"]],
    "fixture_rooms": {"toilet": ["bathroom"], "sink": ["kitchen", "bathroom"],
                      "bath_shower": ["bathroom"], "appliance": ["kitchen"]},
    "room_area": {"living": [300, 5000], "bedroom": [200, 3000], "kitchen": [150, 3000], "bathroom": [100, 2000]},
    "forbidden_door_pairs": [["bathroom", "kitchen"]],
}


def box(lab, r0, r1, c0, c1, cls):
    """Paint rows r0..r1 and cols c0..c1 (inclusive)."""
    lab[r0:r1 + 1, c0:c1 + 1] = I[cls]


def valid_plan():
    lab = np.full((SIZE, SIZE), I["background"], np.uint8)
    box(lab, 11, 45, 11, 68, "living")
    box(lab, 49, 84, 11, 29, "bedroom")
    box(lab, 49, 84, 33, 47, "kitchen")
    box(lab, 49, 84, 51, 68, "bathroom")
    box(lab, 33, 57, 72, 84, "outdoor")
    for r0, r1, c0, c1 in [(8, 10, 8, 71), (85, 87, 8, 71), (8, 87, 8, 10), (8, 87, 69, 71),
                           (46, 48, 11, 68), (49, 84, 30, 32), (49, 84, 48, 50)]:
        box(lab, r0, r1, c0, c1, "wall")
    for r0, r1, c0, c1 in [(30, 32, 72, 87), (58, 60, 72, 87), (30, 60, 85, 87)]:
        box(lab, r0, r1, c0, c1, "railing")
    for r0, r1, c0, c1 in [(46, 48, 14, 25), (46, 48, 35, 44), (46, 48, 54, 63), (8, 10, 40, 51), (34, 44, 69, 71)]:
        box(lab, r0, r1, c0, c1, "door")
    for r0, r1, c0, c1 in [(8, 10, 15, 30), (58, 73, 8, 10), (85, 87, 55, 64)]:
        box(lab, r0, r1, c0, c1, "window")
    box(lab, 70, 76, 63, 68, "toilet")
    box(lab, 60, 72, 51, 56, "bath_shower")
    box(lab, 75, 80, 42, 47, "sink")
    box(lab, 81, 84, 42, 47, "appliance")
    return lab


def hole(r0, r1, c0, c1):
    h = np.zeros((SIZE, SIZE), np.uint8)
    h[r0:r1 + 1, c0:c1 + 1] = 1
    return h


# Holes used by the hole-based constraints (applied to the valid toy as well).
HOLES = {"C2": hole(15, 40, 35, 60), "C5": hole(40, 56, 20, 45), "C17": hole(52, 82, 52, 66)}


def broken(name):
    """(label map, hole or None) that breaks constraint/rule `name`."""
    lab = valid_plan()
    h = HOLES.get(name)
    if name == "C1":
        box(lab, 60, 84, 30, 32, "bedroom")             # interior wall stops short: dead end
    elif name == "C2":
        box(lab, 25, 27, 40, 55, "wall")                # wall piece floating inside the hole
    elif name == "C3":
        for i in range(15):                             # diagonal wall
            lab[15 + i, 38 + i:41 + i] = I["wall"]
    elif name == "C4":
        box(lab, 20, 29, 40, 49, "wall")                # 10 x 10 wall block (too thick)
    elif name == "C5":
        box(lab, 49, 56, 30, 32, "bedroom")             # vertical wall stops at the hole edge instead of continuing
    elif name == "C6":
        box(lab, 60, 70, 30, 32, "bedroom")             # gap in the wall: bedroom touches kitchen
    elif name == "C7":
        box(lab, 20, 20, 40, 55, "bedroom")             # 1-px bedroom sliver inside the living room
    elif name == "C8":
        box(lab, 46, 48, 54, 63, "wall")                # bathroom door walled up: unreachable
    elif name == "C9":
        box(lab, 20, 22, 40, 51, "door")                # door floating in the living room
    elif name == "C10":
        box(lab, 48, 48, 14, 25, "wall")                # door leaves a wall layer: notch
    elif name == "C11":
        box(lab, 49, 55, 35, 44, "sink")                # door opens onto a fixture, not a space
    elif name == "C12":
        box(lab, 46, 48, 35, 44, "window")              # window between two rooms
    elif name == "C13":
        box(lab, 46, 48, 12, 29, "door")                # door 18 px long (max 14)
    elif name == "C14":
        box(lab, 49, 52, 36, 41, "sink")                # sink right behind the kitchen door
    elif name == "C15":
        box(lab, 75, 81, 11, 16, "toilet")              # toilet in the bedroom
    elif name == "C16":
        box(lab, 25, 30, 40, 45, "appliance")           # appliance floating mid-room
    elif name == "C17":
        lab[np.isin(lab, [I["toilet"], I["bath_shower"]])] = I["bathroom"]   # bathroom without fixtures
    elif name == "C18":
        box(lab, 25, 27, 30, 45, "railing")             # railing inside the living room
    elif name == "R1":
        box(lab, 49, 84, 11, 29, "wall")                # bedroom replaced by a tiny one
        box(lab, 50, 57, 12, 25, "bedroom")
        box(lab, 46, 48, 14, 25, "door")
    elif name == "R3":
        box(lab, 73, 82, 48, 50, "door")               # door between kitchen and bathroom
    elif name == "R4":
        box(lab, 8, 10, 40, 51, "wall")                 # front door removed
        box(lab, 34, 44, 69, 71, "window")              # balcony door becomes a window
    else:
        raise KeyError(name)
    return lab, h


# ---------------------------------------------------------------- toys sized by real constants

def real_constant_cases(const):
    """{name: (valid label, broken label)} for the checks whose toy above is too small for the
    REAL constants (configs/stage2/constraint_constants.yaml): C7 (room widths k_c), C13 (door
    lengths up to Lmax_door) and R3 (a pair from the data's forbidden list). Every other
    constraint uses the standard toys with the real constants unchanged."""
    out = {}
    # C7: one living room wide enough for k_living; broken by a living sliver 7 px wide (< k)
    v = np.full((SIZE, SIZE), I["background"], np.uint8)
    box(v, 10, 85, 10, 85, "wall")
    box(v, 13, 82, 13, 82, "living")
    b = v.copy()
    box(b, 30, 60, 86, 92, "living")
    out["C7"] = (v, b)
    # C13: a door one px longer than Lmax_door, in the bottom wall
    v = valid_plan()
    b = v.copy()
    n = min(const["Lmax_door"] + 1, 42)
    box(b, 85, 87, 12, 12 + n - 1, "door")
    out["C13"] = (v, b)
    # R3: a door joining the first forbidden pair the toy can show (living / bedroom / kitchen)
    pairs = [p for p in const.get("forbidden_door_pairs", []) if set(p) <= {"bedroom", "kitchen", "living"}]
    a, c = pairs[0] if pairs else ("bedroom", "bedroom")
    v = valid_plan()
    b = v.copy()
    b[v == I["bedroom"]] = I[a]                         # left room becomes type a
    b[v == I["kitchen"]] = I[c]                         # middle room becomes type c
    box(b, 60, 69, 30, 32, "door")                      # door through the wall between them
    out["R3"] = (v, b)
    return out
