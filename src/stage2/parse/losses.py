"""Parser loss: class-weighted cross-entropy + soft Dice over the classes present in the batch.

CE weights = sqrt(f_max / f_c) (inverse square-root pixel frequency, the most common
class gets 1), capped at 10, so rare classes (doors, fixtures) count more without
exploding. Label 255 is ignored by both terms.
"""

import numpy as np
import torch
import torch.nn.functional as F


def class_weights(freq, cap=10.0):
    """Pixel frequencies (16,) -> CE weights (16,) float tensor."""
    f = np.asarray(freq, float)
    f = np.where(f > 0, f, f[f > 0].min())
    return torch.tensor(np.minimum(np.sqrt(f.max() / f), cap), dtype=torch.float32)


def dice_loss(logits, target, ignore_index=255, eps=1.0):
    """1 - soft Dice, averaged over the classes that occur in the (non-ignored) target."""
    valid = (target != ignore_index)
    p = logits.float().softmax(1) * valid[:, None]
    t = F.one_hot(target.clamp(max=logits.shape[1] - 1), logits.shape[1]).permute(0, 3, 1, 2).float()
    t = t * valid[:, None]
    present = t.sum((0, 2, 3)) > 0
    if not present.any():
        return logits.sum() * 0
    inter = (p * t).sum((0, 2, 3))
    denom = p.sum((0, 2, 3)) + t.sum((0, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice[present].mean()


class ParserLoss(torch.nn.Module):
    def __init__(self, weights, dice=1.0, ignore_index=255):
        super().__init__()
        self.register_buffer("weights", weights)
        self.dice, self.ignore = dice, ignore_index

    def forward(self, logits, target):
        ce = F.cross_entropy(logits.float(), target, weight=self.weights, ignore_index=self.ignore)
        d = dice_loss(logits, target, self.ignore)
        return ce + self.dice * d, {"ce": ce.item(), "dice": d.item()}
