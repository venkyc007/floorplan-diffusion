"""Models: conditional diffusion inpainter and a one-pass U-Net baseline.

Both share one interface used by src/train.py:
    model.loss(cond, target, mask)  -> scalar training loss
    model.predict(cond)             -> predicted label maps (B, H, W)
"""

from src.models.diffusion import DiffusionInpainter
from src.models.unet_baseline import UNetInpainter


def build_model(mcfg):
    """Create the model named by mcfg["kind"] ("diffusion" or "unet")."""
    kinds = {"diffusion": DiffusionInpainter, "unet": UNetInpainter}
    return kinds[mcfg["kind"]](mcfg)
