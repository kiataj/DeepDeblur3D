import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["UNet3D_Residual", "ConvBlock3D"]

def _pick_groups(c: int) -> int:
    """Pick a GroupNorm group count that divides c (8→4→2→1)."""
    for g in (8, 4, 2):
        if c % g == 0:
            return g
    return 1  # fallback: LayerNorm-over-channels behavior

class ConvBlock3D(nn.Module):
    """3D conv block: (Conv3d → GN → LeakyReLU) × 2."""
    def __init__(self, c_in: int, c_out: int, k: int = 3):
        super().__init__()
        p = k // 2
        g = _pick_groups(c_out)
        self.conv1 = nn.Conv3d(c_in, c_out, k, padding=p, bias=False)
        self.gn1   = nn.GroupNorm(g, c_out, affine=True)
        self.act1  = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv3d(c_out, c_out, k, padding=p, bias=False)
        self.gn2   = nn.GroupNorm(g, c_out, affine=True)
        self.act2  = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.gn1(self.conv1(x)))
        x = self.act2(self.gn2(self.conv2(x)))
        return x

class UNet3D_Residual(nn.Module):
    """
    3D U-Net with residual output: y = clamp(x + head(decoder(...)), [0,1]).
    - Safe with AMP (GroupNorm, no InstanceNorm).
    - Strided conv for downsampling, ConvTranspose3d for upsampling.
    """
    def __init__(self, in_ch: int = 1, base: int = 24, levels: int = 4):
        super().__init__()
        assert levels >= 1, "levels must be >= 1"
        chs = [base * (2 ** i) for i in range(levels)]

        self.enc, self.down = nn.ModuleList(), nn.ModuleList()
        c = in_ch
        for co in chs:
            self.enc.append(ConvBlock3D(c_in=c, c_out=co))
            c = co
            self.down.append(nn.Conv3d(c, c, kernel_size=2, stride=2))

        self.bottleneck = ConvBlock3D(c_in=c, c_out=c)

        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        for co in reversed(chs):
            self.up.append(nn.ConvTranspose3d(c, co, kernel_size=2, stride=2))
            self.dec.append(ConvBlock3D(c_in=co * 2, c_out=co))
            c = co

        self.out = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        feats = []
        for e, dwn in zip(self.enc, self.down):
            x = e(x); feats.append(x)
            x = dwn(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.up, self.dec, reversed(feats)):
            x = up(x)
            # match spatial size (handles odd dims)
            dz, dy, dx = skip.shape[2]-x.shape[2], skip.shape[3]-x.shape[3], skip.shape[4]-x.shape[4]
            if dz or dy or dx:
                x = F.pad(x, (0, max(dx, 0), 0, max(dy, 0), 0, max(dz, 0)))
                x = x[:, :, :skip.shape[2], :skip.shape[3], :skip.shape[4]]
            x = dec(torch.cat([x, skip], dim=1))

        y = (x_in + self.out(x)).clamp(0, 1)
        return y
