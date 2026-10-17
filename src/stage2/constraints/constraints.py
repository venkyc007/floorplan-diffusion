"""Stage 2 structural constraints C1-C18 (docs/constraints_spec.md).

Every constraint is ONE function  cN(P, ctx) -> V  that returns a per-pixel
violation map V of shape (B, 1, H, W), values >= 0 (0 = rule satisfied).
The same function serves four uses:
  training loss      constraint_loss(name, P, ctx)  (masked mean over M = hole grown 3 px)
  guidance           gradient of that loss with respect to P (or x0)
  detection          threshold V on a full parsed plan
  verification       count thresholded violations

Inputs
  P    (B, 16, H, W) class probabilities (softmax(x0 / 0.2) for diffusion, softmax(logits)
       for the U-Net, one-hot for ground truth). Compute in fp32.
  ctx  Ctx: the hole, the ground truth Y, the known plan, constants, flood-fill steps.

Modes (REGISTRY[name].mode is the starting default; gate G2 may change it)
  absolute     loss = mean_M V(P)
  gt_hinged    loss = relu(mean_M V(P) - mean_M V(Y))   (only penalise being worse than the real plan)
  gt_relative  V itself uses Y (C1 clDice, C8 reachability in training, C17)

Size constants (px, walls = 3 px) come from configs/stage2/constraint_constants.yaml.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.stage2.constraints import ops
from src.stage2.data.classes16 import C16
from src.utils import PROJECT_ROOT

CONST_FILE = PROJECT_ROOT / "configs" / "stage2" / "constraint_constants.yaml"


def load_constants(path=CONST_FILE):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["values"]


CONST = load_constants()


@dataclass
class Ctx:
    hole: torch.Tensor = None      # (B,1,H,W), 1 = pixel to fill; None = no hole (detection)
    Y: torch.Tensor = None         # (B,16,H,W) ground-truth one-hot (training / gates)
    known: torch.Tensor = None     # (B,16,H,W) known plan, zero in the hole; default Y * (1 - hole)
    seed: torch.Tensor = None      # (B,1,H,W) C8 inference seed; default = largest walkable component
    const: dict = field(default_factory=lambda: CONST)
    K: int = 96                    # flood-fill steps (spec: ~96 for a 128-px window)

    def region(self, like):
        """M = hole grown by 3 px (7x7 square); everywhere if there is no hole."""
        if self.hole is None:
            return torch.ones_like(like[:, :1])
        return ops.dil(self.hole, 7)

    def known_plan(self):
        if self.known is not None:
            return self.known
        if self.Y is not None and self.hole is not None:
            return self.Y * (1 - self.hole)
        raise ValueError("this constraint needs ctx.known or ctx.Y + ctx.hole")


def g(P, group):
    """Sum of the class probabilities of a group (W, D, N, RL, S, OD, BG, OUT, F, PASS, BOUNDARY)."""
    return P[:, C16.group(group)].sum(1, keepdim=True)


def ch(P, name):
    return P[:, C16.index[name]: C16.index[name] + 1]


def frame_like(x):
    """1 on the outermost pixel ring of the image, 0 inside."""
    f = torch.ones_like(x[:, :1])
    f[..., 1:-1, 1:-1] = 0
    return f


# ====================================================================== walls

def c1_wall_ends(P, ctx):
    """C1 (map form): dead ends. The last pixels of a straight wall run (horizontal runs that
    are not part of a vertical run, and vice versa) whose next pixel along the run is not
    wall, door, window, railing, the outside or beyond the image border.

    (Skeleton-free: the soft skeleton of 3-px walls breaks apart at T-junctions and
    corners, which created false end points on valid plans.)
    """
    W = g(P, "W")
    n = ctx.const["T_wall"] + 2
    run_h, run_v = ops.opn_h(W, n), ops.opn_v(W, n)
    only_h, only_v = F.relu(run_h - run_v), F.relu(run_v - run_h)
    # 1 where the walk may legitimately continue; pixels beyond the image count as 1
    ok = (g(P, "BOUNDARY") + g(P, "OUT")).clamp(max=1)
    blocked = 1 - ok
    V = torch.zeros_like(W)
    for run, dy, dx in ((only_h, 0, 1), (only_h, 0, -1), (only_v, 1, 0), (only_v, -1, 0)):
        nxt = ops.shift(blocked, -dy, -dx)          # value of the next pixel in direction (dy, dx)
        V = V + run * nxt
    return V


def cldice_loss(P, ctx, region):
    """C1 training loss: 1 - soft clDice(walls of P, walls of Y) inside the region -> (B,)."""
    t = ctx.const["T_wall"]
    pw, yw = g(P, "W") * region, g(ctx.Y, "W") * region
    sp, sy = ops.soft_skeleton(pw, t + 1), ops.soft_skeleton(yw, t + 1)
    eps = 1e-6
    tprec = ((sp * yw).flatten(1).sum(1) + eps) / (sp.flatten(1).sum(1) + eps)
    tsens = ((sy * pw).flatten(1).sum(1) + eps) / (sy.flatten(1).sum(1) + eps)
    return 1 - 2 * tprec * tsens / (tprec + tsens)


def c2_floating_walls(P, ctx):
    """C2: wall predicted in the hole that is not connected (through boundary pixels) to known structure."""
    if ctx.hole is None:
        return torch.zeros_like(P[:, :1])
    B = g(P, "BOUNDARY").clamp(max=1)
    reach = ops.rec(B * (1 - ctx.hole), B, ctx.K)
    return F.relu(g(P, "W") - reach)


def c3_manhattan(P, ctx):
    """C3: wall pixels on no horizontal or vertical run of >= T_wall + 2 px (diagonal / blobby walls).

    (Line-opening form of the Stage 1 alignment metric; Sobel products also fire at
    every wall end and corner, so they are not ~0 on valid Manhattan plans.)
    """
    W = g(P, "W")
    n = ctx.const["T_wall"] + 2
    return F.relu(W - torch.maximum(ops.opn_h(W, n), ops.opn_v(W, n)))


def c4_thickness(P, ctx):
    """C4: wall too thin (removed by a T_wall opening) or too thick (survives a (T_wall_max+1) erosion)."""
    W = g(P, "W")
    thin = F.relu(W - ops.opn(W, ctx.const["T_wall"]))
    thick = ops.ero(W, ctx.const["T_wall_max"] + 1).clamp(min=0)
    return thin + thick


@torch.no_grad()
def wall_continuations(known_w, hole, t_wall):
    """C5 helper (constant per sample): pixels in the hole that continue a straight known wall,
    up to 2 * T_wall px beyond the hole edge."""
    n = t_wall + 2
    runs = {"h": ops.opn_h(known_w, n), "v": ops.opn_v(known_w, n)}
    open_ = (hole + known_w).clamp(max=1)
    cont = torch.zeros_like(known_w)
    for axis, dy, dx in (("h", 0, 1), ("h", 0, -1), ("v", 1, 0), ("v", -1, 0)):
        c = runs[axis]
        for _ in range(2 * t_wall):
            c = ops.shift(c, dy, dx) * open_
            cont = torch.maximum(cont, c * hole)
    return (cont > 0.5).to(known_w.dtype)


def c5_alignment(P, ctx):
    """C5: inside the hole, the straight continuation of a known wall must be wall, door, window or railing."""
    if ctx.hole is None:
        return torch.zeros_like(P[:, :1])
    known_w = g(ctx.known_plan(), "W")
    cont = wall_continuations(known_w, ctx.hole, ctx.const["T_wall"])
    return cont * F.relu(1 - g(P, "BOUNDARY"))


# ====================================================================== spaces

SPACES8 = C16.group("S") + C16.group("OD") + C16.group("BG")


def contact_matrix(const, device, dtype):
    """8x8 matrix: 1 for pairs of different spaces that must NOT touch directly (C6)."""
    names = [C16.names[i] for i in SPACES8]
    A = 1 - torch.eye(len(names))
    for a, b in const.get("c6_excluded_pairs", []):
        if a in names and b in names:
            A[names.index(a), names.index(b)] = A[names.index(b), names.index(a)] = 0
    return A.to(device=device, dtype=dtype)


def c6_closure(P, ctx):
    """C6: two different spaces directly 4-adjacent (no wall/door/window between them)."""
    S8 = P[:, SPACES8]
    A = contact_matrix(ctx.const, P.device, P.dtype)
    V = torch.zeros_like(P[:, :1])
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        AQ = torch.einsum("ab,nbhw->nahw", A, ops.shift(S8, dy, dx))
        V = V + (S8 * AQ).sum(1, keepdim=True)
    return V


def c7_min_width(P, ctx):
    """C7: parts of a space / fixture class thinner than its minimum width k_c."""
    V = torch.zeros_like(P[:, :1])
    for name, k in ctx.const["k"].items():
        X = ch(P, name)
        V = V + F.relu(X - ops.opn(X, k))
    return V


@torch.no_grad()
def largest_walkable(P):
    """C8 inference seed: the largest 4-connected component of argmax-walkable pixels."""
    walk = (g(P, "PASS") > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)
    seed = np.zeros_like(walk, dtype=np.float32)
    for b in range(walk.shape[0]):
        n, comp, st, _ = cv2.connectedComponentsWithStats(walk[b], connectivity=4)
        if n > 1:
            seed[b] = comp == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return torch.from_numpy(seed).to(P)[:, None]


def c8_reachability(P, ctx):
    """C8: room pixels that cannot be reached by walking (rooms, outdoor, doors).

    With ctx.Y and ctx.hole (training, GT-relative): only rooms the REAL plan reaches from
    the window border count, and the walk starts from those reachable pixels outside the hole.
    Otherwise (inference, absolute): the walk starts from ctx.seed or the largest walkable component.
    """
    walk = g(P, "PASS").clamp(max=1)
    if ctx.Y is not None and ctx.hole is not None:
        walk_y = g(ctx.Y, "PASS").clamp(max=1)
        r_gt = ops.rec(frame_like(P) * (g(ctx.Y, "S") + g(ctx.Y, "OD")), walk_y, ctx.K)
        reach = ops.rec(r_gt * (1 - ctx.hole), walk, ctx.K)
        return g(P, "S") * r_gt * (1 - reach)
    seed = ctx.seed if ctx.seed is not None else largest_walkable(P)
    return g(P, "S") * (1 - ops.rec(seed, walk, ctx.K))


# ====================================================================== openings

def _openings(P, ctx):
    c = ctx.const
    return ((g(P, "D"), c["Lmax_door"], c["Lmin_door"]), (g(P, "N"), c["Lmax_win"], c["Lmin_win"]))


def c9_in_wall(P, ctx):
    """C9: door/window without wall on both ends (along either axis, within its maximum length)."""
    W = g(P, "W")
    V = torch.zeros_like(W)
    for X, lmax, _ in _openings(P, ctx):
        V = V + X * (1 - torch.maximum(ops.flank_h(W, lmax), ops.flank_v(W, lmax)))
    return V


def c10_cut_through(P, ctx):
    """C10: opening that leaves wall on one face (a notch): wall directly beside it across the wall."""
    W = g(P, "W")
    V = torch.zeros_like(W)
    for X, lmax, _ in _openings(P, ctx):
        V = V + X * (ops.flank_h(W, lmax) * ops.sum_v(W) + ops.flank_v(W, lmax) * ops.sum_h(W))
    return V


def c11_door_spaces(P, ctx):
    """C11: a door must have a walkable space (room, outdoor or the outside background) on both faces."""
    k = ctx.const["T_wall"] + 2
    Q = (g(P, "S") + g(P, "OD") + g(P, "BG")).clamp(max=1)
    h = ops.shift(Q, 0, k) * ops.shift(Q, 0, -k)
    v = ops.shift(Q, k, 0) * ops.shift(Q, -k, 0)
    return g(P, "D") * (1 - torch.maximum(h, v))


def c12_window_sides(P, ctx):
    """C12: a window must have an interior space on one face and the outside on the other."""
    k = ctx.const["T_wall"] + 2
    S, O = g(P, "S"), g(P, "OUT")
    h = ops.shift(S, 0, k) * ops.shift(O, 0, -k) + ops.shift(O, 0, k) * ops.shift(S, 0, -k)
    v = ops.shift(S, k, 0) * ops.shift(O, -k, 0) + ops.shift(O, k, 0) * ops.shift(S, -k, 0)
    return g(P, "N") * (1 - torch.maximum(h, v).clamp(max=1))


def c13_length(P, ctx):
    """C13: door/window shorter than its minimum or longer than its maximum length."""
    V = torch.zeros_like(P[:, :1])
    for X, lmax, lmin in _openings(P, ctx):
        short = F.relu(X - torch.maximum(ops.opn_h(X, lmin), ops.opn_v(X, lmin)))
        long_ = torch.maximum(ops.ero_h(X, lmax + 1), ops.ero_v(X, lmax + 1)).clamp(min=0)
        V = V + short + long_
    return V


def c14_clearance(P, ctx):
    """C14: fixture within the door clearance distance."""
    return ops.dil(g(P, "D"), ctx.const["k_clear"]) * g(P, "F")


# ====================================================================== fixtures

def c15_fixture_room(P, ctx):
    """C15: fixture not inside (or next to) one of its compatible room types A(f)."""
    V = torch.zeros_like(P[:, :1])
    for f, rooms in ctx.const["fixture_rooms"].items():
        allowed = sum(ch(P, r) for r in rooms) if rooms else torch.zeros_like(V)
        V = V + ch(P, f) * (1 - ops.dil(allowed.clamp(max=1), ctx.const["k_fix"]))
    return V


def c16_against_wall(P, ctx):
    """C16: fixture blob that touches no wall/window/railing."""
    Fx = g(P, "F").clamp(max=1)
    support = ops.dil((g(P, "W") + g(P, "N") + g(P, "RL")).clamp(max=1), 3)
    return F.relu(Fx - ops.rec(Fx * support, Fx, 32))


REQUIRED = {"bathroom": ("toilet", "bath_shower"), "kitchen": ("sink", "appliance")}


def _required_instances(Y, hole):
    """GT bathroom / kitchen instances fully inside the window that overlap the hole.

    An instance = a 4-connected region of that room type after every fixture pixel took the
    type of the nearest room pixel (so the fixtures standing in the room belong to it; same
    definition as R1, user rule 2026-10-17).
    Returns [(sample index, room name, bool mask tensor (H, W))].
    """
    from src.stage2.data.plan_stats import fill_fixtures
    out = []
    lab = Y.argmax(1).cpu().numpy()
    hl = hole[:, 0].cpu().numpy() > 0.5 if hole is not None else np.ones(lab.shape, bool)
    for b in range(lab.shape[0]):
        filled = fill_fixtures(lab[b])
        for room in REQUIRED:
            n, comp = cv2.connectedComponents((filled == C16.index[room]).astype(np.uint8), connectivity=4)
            for i in range(1, n):
                m = comp == i
                if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any() or not (m & hl[b]).any():
                    continue
                out.append((b, room, torch.from_numpy(m).to(Y.device)))
    return out


def c17_required_fixtures(P, ctx):
    """C17 (training only, GT-hinged per instance): a bathroom needs a toilet or bath/shower, a kitchen a
    sink or appliance. For each qualifying GT instance: relu(smax(required in Y) - smax(required in P)),
    spread over the instance's pixels."""
    V = torch.zeros_like(P[:, :1])
    if ctx.Y is None:
        return V
    for b, room, m in _required_instances(ctx.Y, ctx.hole):
        need = REQUIRED[room]
        p = sum(ch(P[b:b + 1], f) for f in need)[0, 0]
        y = sum(ch(ctx.Y[b:b + 1], f) for f in need)[0, 0]
        gap = F.relu(ops.smax(y[None], m[None]) - ops.smax(p[None], m[None]))[0]
        V[b, 0] = V[b, 0] + m.to(V.dtype) * gap
    return V


def c17_loss(P, ctx):
    """C17 as a loss: mean over qualifying instances of the per-instance gap -> (B,)."""
    loss = torch.zeros(P.shape[0], device=P.device, dtype=P.dtype)
    count = torch.zeros_like(loss)
    if ctx.Y is None:
        return loss
    for b, room, m in _required_instances(ctx.Y, ctx.hole):
        need = REQUIRED[room]
        p = sum(ch(P[b:b + 1], f) for f in need)[0, 0]
        y = sum(ch(ctx.Y[b:b + 1], f) for f in need)[0, 0]
        loss[b] = loss[b] + F.relu(ops.smax(y[None], m[None]) - ops.smax(p[None], m[None]))[0]
        count[b] += 1
    return loss / count.clamp(min=1)


# ====================================================================== exterior

def c18_railing(P, ctx):
    """C18: railing with no outdoor space within one wall thickness (2*T_wall+1 square)."""
    return g(P, "RL") * (1 - ops.dil(g(P, "OD"), 2 * ctx.const["T_wall"] + 1))


# ====================================================================== registry + loss

@dataclass
class Spec:
    fn: object
    mode: str              # default mode: absolute | gt_hinged | gt_relative
    needs_hole: bool = False
    training_only: bool = False
    title: str = ""


REGISTRY = {
    "C1": Spec(c1_wall_ends, "gt_relative", title="walls connect (no dead ends)"),
    "C2": Spec(c2_floating_walls, "absolute", needs_hole=True, title="no floating wall pieces"),
    "C3": Spec(c3_manhattan, "gt_hinged", title="walls axis-aligned"),
    "C4": Spec(c4_thickness, "gt_hinged", title="wall thickness in range"),
    "C5": Spec(c5_alignment, "absolute", needs_hole=True, title="walls continue into the hole"),
    "C6": Spec(c6_closure, "absolute", title="different spaces never touch"),
    "C7": Spec(c7_min_width, "gt_hinged", title="minimum width per space/fixture"),
    "C8": Spec(c8_reachability, "gt_relative", title="every room reachable through doors"),
    "C9": Spec(c9_in_wall, "absolute", title="openings sit in a wall"),
    "C10": Spec(c10_cut_through, "gt_hinged", title="openings cut through the wall"),
    "C11": Spec(c11_door_spaces, "absolute", title="door joins two spaces"),
    "C12": Spec(c12_window_sides, "absolute", title="window: inside one side, outside other"),
    "C13": Spec(c13_length, "gt_hinged", title="opening length in range"),
    "C14": Spec(c14_clearance, "gt_hinged", title="door clearance"),   # user: GT-hinged (2026-10-17)
    "C15": Spec(c15_fixture_room, "absolute", title="fixture in a compatible room"),
    "C16": Spec(c16_against_wall, "gt_hinged", title="fixture against a wall"),
    "C17": Spec(c17_required_fixtures, "gt_relative", training_only=True, title="required fixtures"),
    "C18": Spec(c18_railing, "absolute", title="railing borders outdoor"),
}


def violation_map(name, P, ctx):
    """V for one constraint, always computed in fp32."""
    with torch.autocast(device_type=P.device.type, enabled=False):
        return REGISTRY[name].fn(P.float(), ctx)


def constraint_loss(name, P, ctx, mode=None):
    """Masked-mean loss of one constraint -> (B,). mode: absolute | gt_hinged | gt_relative (default: registry)."""
    mode = mode or REGISTRY[name].mode
    P = P.float()
    with torch.autocast(device_type=P.device.type, enabled=False):
        M = ctx.region(P)
        if name == "C1" and mode == "gt_relative":
            return cldice_loss(P, ctx, M)
        if name == "C17":
            return c17_loss(P, ctx)
        V = REGISTRY[name].fn(P, ctx)
        loss = ops.masked_mean(V, M)
        if mode == "gt_hinged":
            loss = F.relu(loss - ops.masked_mean(REGISTRY[name].fn(ctx.Y.float(), ctx), M))
        return loss


def probabilities(x0, temperature=0.2):
    """Diffusion x0 prediction in [-1, 1] -> class probabilities (spec: tau = 0.2)."""
    return F.softmax(x0.float() / temperature, dim=1)


def one_hot(label, n=C16.n):
    """(B, H, W) int labels -> (B, n, H, W) float one-hot."""
    return F.one_hot(label.long(), n).permute(0, 3, 1, 2).float()
