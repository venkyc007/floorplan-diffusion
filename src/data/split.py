"""Splitting plans into train / val / test WITHOUT leakage.

We never split individual plans. We split GROUPS of similar plans
(from dedup.py): a whole group always goes into the same split, so two
similar plans can never end up on different sides.
"""

import random
from collections import defaultdict


def split_groups(group_of, fractions, seed):
    """Assign every plan to 'train', 'val' or 'test'.

    group_of:  list, entry i = group number of plan i
    fractions: e.g. {"train": 0.8, "val": 0.1, "test": 0.1}
    seed:      random seed, so the result is the same on every run

    Returns a list, entry i = split name of plan i.
    """
    members = defaultdict(list)
    for plan, group in enumerate(group_of):
        members[group].append(plan)

    groups = list(members.values())
    rng = random.Random(seed)
    rng.shuffle(groups)
    # Place big groups first (while there is room everywhere), small ones
    # last to fine-tune the sizes. The shuffle above randomizes ties.
    groups.sort(key=len, reverse=True)

    n_total = len(group_of)
    target = {name: frac * n_total for name, frac in fractions.items()}
    count = {name: 0 for name in fractions}
    split_of = [None] * n_total
    for group in groups:
        # Put the group in the split that is furthest below its target size.
        name = max(fractions, key=lambda s: target[s] - count[s])
        for plan in group:
            split_of[plan] = name
        count[name] += len(group)
    return split_of
