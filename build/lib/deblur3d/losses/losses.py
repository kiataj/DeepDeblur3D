import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- SSIM 3D ----------
@torch.no_grad()
def _box_kernel_3d(win: int, device):
    w = torch.ones(1, 1, win, win, win, device=device, dtype=torch.float32)
    w /= (win ** 3)
    return w

def ssim3d(x: torch.Tensor, y: torch.Tensor, C1: float = 1e-4, C2: float = 9e-4, win: int = 7) -> torch.Tensor:
    """
    3D SSIM for single-channel volumes.
    Args:
        x, y: (N,1,D,H,W) or (1,1,D,H,W) float32 in [0,1]
        C1, C2: SSIM constants (for [0,1] data)
        win: cubic window size
    Returns:
        mean SSIM over the volume (scalar tensor)
    """
    assert x.shape == y.shape and x.dim() == 5 and x.size(1) == 1, "Expect (N,1,D,H,W)"
    pad = win // 2
    w = _box_kernel_3d(win, x.device)

    mu_x = F.conv3d(x, w, padding=pad)
    mu_y = F.conv3d(y, w, padding=pad)
    sig_x = F.conv3d(x * x, w, padding=pad) - mu_x * mu_x
    sig_y = F.conv3d(y * y, w, padding=pad) - mu_y * mu_y
    sig_xy = F.conv3d(x * y, w, padding=pad) - mu_x * mu_y

    num  = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den  = (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x + sig_y + C2) + 1e-12
    ssim = (num / den).clamp(min=-1, max=1)
    return ssim.mean()

# ---------- Frequency losses ----------
def freq_l1(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    L1 in Fourier magnitude; emphasizes high-freqs when alpha>1.
    x,y: (N,1,D,H,W)
    """
    X = torch.fft.fftn(x, dim=(-3, -2, -1)).abs()
    Y = torch.fft.fftn(y, dim=(-3, -2, -1)).abs()
    return (X - Y).abs().pow(alpha).mean()

def freq_l1_relative(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Relative spectrum difference: | |X|-|Y| | / (|X|+|Y|+eps).
    More scale-stable than raw L1 on magnitudes.
    """
    X = torch.fft.fftn(x, dim=(-3, -2, -1)).abs()
    Y = torch.fft.fftn(y, dim=(-3, -2, -1)).abs()
    return ((X - Y).abs() / (X + Y + eps)).mean()

# ---------- Combined loss ----------
class DeblurLoss(nn.Module):
    """
    L = w_l1 * L1 + w_ssim*(1-SSIM3D) + w_freq*FreqLoss  + id_weight*L1(pred, blurred)
    Use freq_l1_relative by swapping the line in forward().
    """
    def __init__(self, w_l1: float = 0.8, w_ssim: float = 0.2, w_freq: float = 0.05, id_weight: float = 0.1,
                 use_relative_freq: bool = False, freq_alpha: float = 1.0):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_freq = w_freq
        self.idw = id_weight
        self.use_relative_freq = use_relative_freq
        self.freq_alpha = freq_alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor, blurred: torch.Tensor = None) -> torch.Tensor:
        # ensure (N,1,D,H,W)
        if pred.dim() == 4: pred = pred.unsqueeze(1)
        if target.dim() == 4: target = target.unsqueeze(1)
        l = self.w_l1 * F.l1_loss(pred, target)
        l += self.w_ssim * (1.0 - ssim3d(pred, target))
        if self.w_freq > 0.0:
            if self.use_relative_freq:
                l += self.w_freq * freq_l1_relative(pred, target)
            else:
                l += self.w_freq * freq_l1(pred, target, alpha=self.freq_alpha)
        if blurred is not None and self.idw > 0.0:
            if blurred.dim() == 4: blurred = blurred.unsqueeze(1)
            l += self.idw * F.l1_loss(pred, blurred)
        return l
