"""The U-Net "backbone" shared by both models.

A U-Net shrinks the image step by step (128 -> 64 -> 32 -> 16 px) to
understand the big picture, then grows it back to full size to draw the
details, with "skip connections" passing fine details across. We use the
ready-made UNet2DModel from HuggingFace diffusers.
"""

from diffusers import UNet2DModel


def build_unet(mcfg, in_channels, out_channels, sample_size=128):
    """Build a UNet2DModel from the `model:` section of a config.

    Self-attention (lets every pixel look at every other pixel; costly) is
    used only at levels >= attention_from_level, i.e. the small 16x16 maps,
    plus the middle block.
    """
    channels = tuple(mcfg["block_out_channels"])
    n, first = len(channels), mcfg["attention_from_level"]
    down = ["AttnDownBlock2D" if level >= first else "DownBlock2D" for level in range(n)]
    up = list(reversed([b.replace("Down", "Up") for b in down]))
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        block_out_channels=channels,
        layers_per_block=mcfg["layers_per_block"],
        down_block_types=down,
        up_block_types=up,
        attention_head_dim=mcfg["attention_head_dim"],
        norm_num_groups=mcfg.get("norm_num_groups", 32),  # tiny test models need fewer groups
    )


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())
