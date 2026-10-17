"""The 16 Stage 2 classes, read once from configs/stage2/classes16.yaml.

Everything else imports class indices, colours and groups from here, so the
order is defined in exactly one place.

    from src.stage2.data.classes16 import C16
    C16.index["wall"]        -> 0
    C16.group("S")           -> [4, 5, 6, 7, 8, 9]   (indices of all room types)
    C16.palette              -> (16, 3) uint8 colours for figures
"""

from functools import cached_property

import numpy as np
import yaml

from src.utils import PROJECT_ROOT

CONFIG = PROJECT_ROOT / "configs" / "stage2" / "classes16.yaml"


class Classes16:
    def __init__(self, path=CONFIG):
        with open(path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.names = list(self.cfg["classes"])
        self.index = {n: i for i, n in enumerate(self.names)}
        self.n = len(self.names)

    def group(self, name):
        """Class indices of a constraint group (W, D, N, RL, S, OD, BG, OUT, F, PASS, BOUNDARY)."""
        return [self.index[c] for c in self.cfg["groups"][name]]

    @cached_property
    def palette(self):
        return np.array([self.cfg["colours"][n] for n in self.names], dtype=np.uint8)

    def space_class(self, space_type):
        """Class index for a CubiCasa 'Space <Type>' (unknown types -> space_default)."""
        name = self.cfg["spaces"].get(space_type, self.cfg["space_default"])
        return self.index[name]

    def fixture_class(self, class_words):
        """Class index for 'FixedFurniture <Name> [Subtype]' words, or None if ignored.

        class_words = the element's class attribute split on spaces, without
        'FixedFurniture'. A listed sub-type override (e.g. SaunaStove) wins.
        """
        table = self.cfg["fixtures"]
        name = class_words[0] if class_words else "Misc"
        for sub in class_words[1:]:
            if sub in self.cfg["fixture_subtypes_override"]:
                name = sub
        target = table.get(name, "ignore")
        return None if target == "ignore" else self.index[target]

    def colourise(self, label):
        """(H, W) label map -> (H, W, 3) uint8 RGB image. 255 (unknown) -> grey."""
        lut = np.full((256, 3), 128, dtype=np.uint8)
        lut[: self.n] = self.palette
        return lut[label]


C16 = Classes16()
