import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ControlledUNet3D", "gaussian_blur3d"]

@torch.no_grad()
def _gauss1d(sigma: float, device, dtype=torch.float32, radius_mult: float = 3.0):
    sigma = max(1e-6, float(sigma))
    r = int(math.ceil(radius_mult * sigma))
    r = max(r, 1)
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / (k.sum() + 1e-12)
    return k, r

@torch.no_grad()
def gaussian_blur3d(x: torch.Tensor, sigma: float, pad_mode: str = "replicate") -> torch.Tensor:
    """
    Separable 3D Gaussian blur for a 5D tensor (N,C,D,H,W).
    """
    if sigma <= 0:
        return x
    k1d, r = _gauss1d(sigma, x.device, x.dtype)
    kz = k1d.view(1, 1, -1, 1, 1)
    ky = k1d.view(1, 1, 1, -1, 1)
    kx = k1d.view(1, 1, 1, 1, -1)
    y = F.conv3d(F.pad(x, (0, 0, 0, 0, r, r), mode=pad_mode), kz, padding=0, groups=x.shape[1])
    y = F.conv3d(F.pad(y, (0, 0, r, r, 0, 0), mode=pad_mode), ky, padding=0, groups=x.shape[1])
    y = F.conv3d(F.pad(y, (r, r, 0, 0, 0, 0), mode=pad_mode), kx, padding=0, groups=x.shape[1])
    return y

class ControlledUNet3D(nn.Module):
    """
    Wraps a residual UNet (predicts y ≈ clamp(x + resid)) and exposes inference-time controls:
      - strength: global residual scale (0=identity, 1=original, >1=stronger)
      - hp_sigma: Gaussian sigma (vox) to split residual into low/high freq
      - hp_gain/lp_gain: scale high/low frequency parts of the residual
    """
    def __init__(self, net: nn.Module, clamp01: bool = True):
        super().__init__()
        self.net = net
        self.clamp01 = clamp01

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        strength: float = 1.0,
        hp_sigma: float = 0.0,
        hp_gain: float = 1.0,
        lp_gain: float = 1.0,
    ) -> torch.Tensor:
        """
        x: (N,1,D,H,W) in [0,1]
        """
        # Get the model's deblur/denoise output and residual
        y = self.net(x)                      # (N,1,D,H,W)
        r = y - x                            # residual the model wants to add

        # Optional frequency split of residual
        if hp_sigma and hp_sigma > 0:
            r_lp = gaussian_blur3d(r, sigma=hp_sigma)   # low frequencies (base/form)
            r_hp = r - r_lp                              # high frequencies (edges/noise)
            r_mod = lp_gain * r_lp + hp_gain * r_hp
        else:
            r_mod = r

        # Global intensity of the correction
        y_ctrl = x + float(strength) * r_mod
        if self.clamp01:
            y_ctrl = y_ctrl.clamp(0, 1)
        return y_ctrl
