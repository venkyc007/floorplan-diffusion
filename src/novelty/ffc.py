"""Image-wide context for the large-hole models: residual Fourier blocks (after LaMa, Suvorov et al., WACV 2022).

A normal convolution only mixes a pixel with its 3 x 3 neighbours, so information from the far side of a large
hole needs many layers to arrive. A Fourier unit transforms the feature map with a 2-D FFT, applies a 1 x 1
convolution to the spectrum and transforms back: every output pixel then depends on EVERY input pixel in one
step (one frequency = one image-wide stripe pattern, which also suits the repetitive, axis-aligned structure of
walls).

    spectral = SpectralAdapters(unet, at=("down2", "mid", "up0"))

adds one residual block  x -> x + gate(FourierUnit(x))  after the chosen blocks of a diffusers UNet2DModel
(down2 = the deepest down block, mid = the middle block, up0 = the first up block). The block's last layer starts
at ZERO, so a warm-started network computes exactly what it did before and learns to use the new path gradually.
The blocks are attached with forward hooks, so the UNet's own weights keep their names (B4 checkpoints load).
Note: the hooks point at THIS object, so never copy.deepcopy a model that has adapters; build a second model and
load the state dict instead (src/novelty/models.py does that for the EMA copy).
"""

import torch
import torch.nn as nn


class FourierUnit(nn.Module):
    def __init__(self, ch, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(2 * ch, 2 * ch, 1, bias=False)
        self.norm = nn.GroupNorm(groups, 2 * ch)
        self.act = nn.SiLU()

    def forward(self, x):
        h, w = x.shape[-2:]
        f = torch.fft.rfft2(x.float(), norm="ortho")              # complex (B, C, H, W // 2 + 1); fp32 for the FFT
        f = self.act(self.norm(self.conv(torch.cat([f.real, f.imag], 1)).float()))
        re, im = f.chunk(2, 1)
        return torch.fft.irfft2(torch.complex(re, im), s=(h, w), norm="ortho")


class SpectralBlock(nn.Module):
    """x + out(h + FourierUnit(h)), h = in(x) at half the channels; `out` starts at zero (identity at first)."""

    def __init__(self, ch, groups=8):
        super().__init__()
        mid = ch // 2
        self.inp = nn.Sequential(nn.GroupNorm(groups, ch), nn.SiLU(), nn.Conv2d(ch, mid, 1))
        self.fu = FourierUnit(mid, groups)
        self.out = nn.Conv2d(mid, ch, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h = self.inp(x).float()
        return x + self.out(h + self.fu(h)).to(x.dtype)


class SpectralAdapters(nn.Module):
    """Spectral blocks hooked after blocks of a UNet2DModel. `at`: names among down<i>, mid, up<i>."""

    def __init__(self, unet, at=("down2", "mid", "up0"), groups=8):
        super().__init__()
        self.at = list(at)
        self.blocks = nn.ModuleDict()
        for name in self.at:
            block = self._block(unet, name)
            ch = block.resnets[-1].conv2.out_channels
            self.blocks[name] = SpectralBlock(ch, groups)
            block.register_forward_hook(self._hook(name))

    @staticmethod
    def _block(unet, name):
        if name == "mid":
            return unet.mid_block
        return (unet.down_blocks if name.startswith("down") else unet.up_blocks)[int(name[-1])]

    def _hook(self, name):
        def hook(module, inputs, output):
            block = self.blocks[name]
            if isinstance(output, tuple):                    # down block: (hidden, skip states); the last skip
                h = block(output[0])                         # state is the same tensor as hidden
                return h, tuple(output[1][:-1]) + (h,)
            return block(output)
        return hook
