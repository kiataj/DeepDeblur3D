import math
import torch
import torch.nn.functional as F

__all__ = ["gaussian_iso_blur3d_cpu", "GaussianIsoBlurCPUTransform"]

def _fwhm_to_sigma(f):  # FWHM (vox) -> sigma (vox)
    return max(0.0, float(f)) / 2.35482004503

def _gauss1d_kernel_from_fwhm(fwhm_vox, radius_mult=3.0, min_sigma=1e-4):
    sigma = max(min_sigma, _fwhm_to_sigma(fwhm_vox))
    r = int(math.ceil(radius_mult * sigma))
    r = max(r, 1)
    x = torch.arange(-r, r + 1, dtype=torch.float32)  # CPU
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = (k / (k.sum() + 1e-12)).to(torch.float32)
    return k, r

@torch.no_grad()
def gaussian_iso_blur3d_cpu(
    vol,                    # (D,H,W) torch.float32 on CPU in [0,1]
    fwhm_vox,               # isotropic FWHM in voxels (float)
    radius_mult=3.0,
    pad_mode="replicate",   # "replicate" | "reflect" | "constant"
    add_poisson=False, poisson_gain=1000.0,
    add_readnoise=False, read_noise_std=0.003,
    clamp01=True,
):
    """
    Separable 3D Gaussian (same FWHM along z,y,x), CPU-only.
    """
    assert vol.device.type == "cpu", "CPU-only transform"
    assert vol.dim() == 3 and vol.dtype == torch.float32
    D, H, W = vol.shape

    k1d, r = _gauss1d_kernel_from_fwhm(fwhm_vox, radius_mult=radius_mult)

    x = vol.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)

    # z-pass
    kz = k1d.view(1, 1, -1, 1, 1)
    x = F.conv3d(F.pad(x, (0, 0, 0, 0, r, r), mode=pad_mode), kz, padding=0)

    # y-pass
    ky = k1d.view(1, 1, 1, -1, 1)
    x  = F.conv3d(F.pad(x, (0, 0, r, r, 0, 0), mode=pad_mode), ky, padding=0)

    # x-pass
    kx = k1d.view(1, 1, 1, 1, -1)
    x  = F.conv3d(F.pad(x, (r, r, 0, 0, 0, 0), mode=pad_mode), kx, padding=0)

    y = x.squeeze(0).squeeze(0).contiguous()  # (D,H,W) on CPU

    if add_poisson:
        g = float(poisson_gain)
        y = torch.poisson(y.clamp(0, 1) * g) / g
    if add_readnoise:
        y = y + torch.randn_like(y) * float(read_noise_std)
    if clamp01:
        y.clamp_(0, 1)
    return y

class GaussianIsoBlurCPUTransform:
    """
    CPU-only isotropic Gaussian blur (optional noise). If fwhm_range is a tuple, samples per call.
    Designed to run inside DataLoader workers (num_workers>0).
    """
    def __init__(
        self,
        fwhm_range=1.5,          # float (fixed) or (min,max) in voxels
        radius_mult=3.0,
        pad_mode="replicate",
        add_noise=False,
        poisson_gain_range=(600.0, 1600.0),
        read_noise_std_range=(0.002, 0.008),
        clamp01=True,
    ):
        self.fwhm_range = fwhm_range
        self.radius_mult = radius_mult
        self.pad_mode = pad_mode
        self.add_noise = add_noise
        self.poisson_gain_range = poisson_gain_range
        self.read_noise_std_range = read_noise_std_range
        self.clamp01 = clamp01

    def _sample(self, v):
        if isinstance(v, (tuple, list)):
            a, b = float(v[0]), float(v[1])
            return float(torch.empty(()).uniform_(a, b).item())
        return float(v)

    def __call__(self, vol_cpu_float01: torch.Tensor) -> torch.Tensor:
        assert vol_cpu_float01.device.type == "cpu"
        v = vol_cpu_float01.contiguous()
        fwhm = self._sample(self.fwhm_range)
        if self.add_noise:
            gain = self._sample(self.poisson_gain_range)
            rstd = self._sample(self.read_noise_std_range)
        else:
            gain, rstd = 1000.0, 0.0
        y = gaussian_iso_blur3d_cpu(
            v, fwhm_vox=fwhm, radius_mult=self.radius_mult, pad_mode=self.pad_mode,
            add_poisson=self.add_noise, poisson_gain=gain,
            add_readnoise=self.add_noise, read_noise_std=rstd,
            clamp01=self.clamp01,
        )
        return y
