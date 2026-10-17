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

    def fixture_target(self, class_words):
        """'FixedFurniture <Name> [Subtype ...]' words (without 'FixedFurniture') ->
        (target, key): target is a class name, 'ignore' or 'appliance_candidate';
        key is the appliance sub-type (the word after ElectricalAppliance, else the name).
        """
        name = class_words[0] if class_words else "Misc"
        if any(w.startswith(p) for w in class_words for p in self.cfg["fixture_ignore_prefixes"]):
            return "ignore", name
        target = self.cfg["fixtures"].get(name, "ignore")
        key = class_words[1] if name == "ElectricalAppliance" and len(class_words) > 1 else name
        return target, key

    def fixture_class(self, class_words, appliance_subtypes):
        """Class index for fixture words, or None if not painted.

        appliance_subtypes: set of sub-type keys that count as appliances, or "all"
        (survey pass: every candidate is painted as an appliance).
        """
        target, key = self.fixture_target(class_words)
        if target == "appliance_candidate":
            target = "appliance" if appliance_subtypes == "all" or key in appliance_subtypes else "ignore"
        return None if target == "ignore" else self.index[target]

    @cached_property
    def appliance_subtypes(self):
        """Appliance sub-types chosen by the survey (configs/stage2/appliance_subtypes.yaml)."""
        path = CONFIG.parent / "appliance_subtypes.yaml"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run python -m src.stage2.data.survey first")
        with open(path, encoding="utf-8") as f:
            return set(yaml.safe_load(f)["appliance"])

    def colourise(self, label):
        """(H, W) label map -> (H, W, 3) uint8 RGB image. 255 (unknown) -> grey."""
        lut = np.full((256, 3), 128, dtype=np.uint8)
        lut[: self.n] = self.palette
        return lut[label]


C16 = Classes16()
