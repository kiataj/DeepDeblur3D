# src/deblur3d/transforms/lamino_blur.py
import math
import torch
import torch.nn.functional as F

__all__ = [
    "lamino_blur_lorentz",
    "LorentzLaminoBlurTransform",
]

def _fwhm_to_sigma(f): 
    return max(0.0, float(f)) / 2.35482004503

def _ensure_xy_pair(val):
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return float(val[0]), float(val[1])
    v = float(val); return v, v

def _gauss2d_kernel_from_fwhm(fwhm_xy, device, dtype, radius_mult=3.0):
    fx, fy = _ensure_xy_pair(fwhm_xy)
    sx, sy = _fwhm_to_sigma(fx), _fwhm_to_sigma(fy)
    if sx <= 1e-6 and sy <= 1e-6:
        return None, (0,0)
    rx = int(math.ceil(radius_mult * max(sx, 1e-6)))
    ry = int(math.ceil(radius_mult * max(sy, 1e-6)))
    x = torch.arange(-rx, rx+1, device=device, dtype=dtype)
    y = torch.arange(-ry, ry+1, device=device, dtype=dtype)
    gx = torch.exp(-(x*x)/(2*sx*sx)); gx /= (gx.sum()+1e-12)
    gy = torch.exp(-(y*y)/(2*sy*sy)); gy /= (gy.sum()+1e-12)
    k = gy[:,None] @ gx[None,:]
    k /= (k.sum()+1e-12)
    return k, (ry, rx)

def _lorentz2d_kernel_from_fwhm(fwhmL_xy, device, dtype, tail_mult=15, max_r=None):
    fx, fy = _ensure_xy_pair(fwhmL_xy)
    gx, gy = max(1e-4, fx*0.5), max(1e-4, fy*0.5)
    rx = int(math.ceil(tail_mult * gx))
    ry = int(math.ceil(tail_mult * gy))
    if max_r is not None:
        rx = min(rx, int(max_r)); ry = min(ry, int(max_r))
    rx, ry = max(1, rx), max(1, ry)
    x = torch.arange(-rx, rx+1, device=device, dtype=dtype)
    y = torch.arange(-ry, ry+1, device=device, dtype=dtype)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    L = 1.0 / (1.0 + (X/gx)**2 + (Y/gy)**2)
    L = L / (L.sum() + 1e-12)
    return L, (ry, rx)

def _lorentz1d_kernel_from_fwhm(FL, device, dtype, tail_mult=20, max_r=None):
    FL = max(1e-4, float(FL)); gamma = FL * 0.5
    r = int(math.ceil(tail_mult * gamma))
    if max_r is not None: r = min(r, int(max_r))
    r = max(r, 1)
    z = torch.arange(-r, r+1, device=device, dtype=dtype)
    L = 1.0 / (1.0 + (z/gamma)**2)
    L = L / (L.sum() + 1e-12)
    return L.view(1,1,-1,1,1), r  # (1,1,k,1,1), support

@torch.no_grad()
def lamino_blur_lorentz(
    vol,
    xy_mode="none",           # 'none' | 'gaussian' | 'lorentz'
    fwhm_xy=0.0,              # scalar or (fx,fy) for Gaussian FWHM (vox)
    fwhmL_xy=0.0,             # scalar or (fx,fy) for Lorentzian FWHM (vox)
    radius_mult_xy=3.0,
    tail_mult_xy=15,
    fwhmL_z_min=2.0,
    fwhmL_z_max=16.0,
    p=1.5,
    z0=None,
    K=12,
    tail_mult_z=20,
    cap_frac=0.5,
    pad_mode_2d="replicate",
    pad_mode_3d="replicate",
    wedge_alpha_deg=None, wedge_soft_beta=2.0,
    add_poisson=False, poisson_gain_range=(300.0,1500.0),
    add_readnoise=False, read_noise_std_range=(0.003,0.008),
    clamp01=False
):
    assert vol.dim()==3 and vol.dtype==torch.float32
    device, dt = vol.device, vol.dtype
    D,H,W = vol.shape

    # (1) XY blur
    if xy_mode == "gaussian" and (isinstance(fwhm_xy, (tuple, list)) or float(fwhm_xy) > 0):
        k2, (ph, pw) = _gauss2d_kernel_from_fwhm(fwhm_xy, device, dt, radius_mult=radius_mult_xy)
        x2 = F.conv2d(F.pad(vol.view(D,1,H,W), (pw,pw,ph,ph), mode=pad_mode_2d),
                      k2.view(1,1,*k2.shape)).view(D,H,W)
    elif xy_mode == "lorentz" and (isinstance(fwhmL_xy, (tuple, list)) or float(fwhmL_xy) > 0):
        k2, (ph, pw) = _lorentz2d_kernel_from_fwhm(fwhmL_xy, device, dt, tail_mult=tail_mult_xy)
        x2 = F.conv2d(F.pad(vol.view(D,1,H,W), (pw,pw,ph,ph), mode=pad_mode_2d),
                      k2.view(1,1,*k2.shape)).view(D,H,W)
    else:
        x2 = vol

    # (2) depth-variant Lorentzian along z
    z = torch.arange(D, device=device, dtype=dt)
    zc = (D-1)/2.0 if z0 is None else torch.tensor(float(z0), device=device, dtype=dt)
    nz = (z - zc).abs() / max(0.5*(D-1), 1e-6)
    FLmin, FLmax = float(fwhmL_z_min), float(fwhmL_z_max)
    centers = torch.linspace(0,1,steps=K, device=device, dtype=dt)
    FL_bins = FLmin + (FLmax - FLmin) * (centers ** p)

    X = x2.unsqueeze(0).unsqueeze(0)
    max_r_cap = cap_frac * (D/2.0)
    outs = []
    for FL in FL_bins.tolist():
        k1, r = _lorentz1d_kernel_from_fwhm(FL, device, dt, tail_mult=tail_mult_z, max_r=max_r_cap)
        Y = F.conv3d(F.pad(X, (0,0,0,0,r,r), mode=pad_mode_3d), k1, padding=0)
        outs.append(Y)
    O = torch.cat(outs, dim=1)

    pos = nz * (K - 1)
    i0 = torch.clamp(pos.floor().long(), 0, K-1)
    i1 = torch.clamp(i0 + 1,           0, K-1)
    t  = (pos - i0.float()).clamp(0, 1)

    Wt = torch.zeros(1, K, D, 1, 1, device=device, dtype=dt)
    ar = torch.arange(D, device=device)
    Wt[0, i0, ar, 0, 0] = 1.0 - t
    Wt[0, i1, ar, 0, 0] += t

    y = (O * Wt).sum(dim=1).squeeze(0)

    # (3) optional wedge
    if wedge_alpha_deg is not None:
        ky = torch.fft.fftfreq(H, d=1.0).to(device)
        kx = torch.fft.fftfreq(W, d=1.0).to(device)
        kz = torch.fft.fftfreq(D, d=1.0).to(device)
        Kz, Ky, Kx = torch.meshgrid(kz, ky, kx, indexing="ij")
        rho_xy = torch.sqrt(Kx*Kx + Ky*Ky) + 1e-12
        theta = torch.atan2(Kz.abs(), rho_xy)
        a = math.radians(float(wedge_alpha_deg))
        delta = torch.clamp(theta - a, min=0.0)
        M = torch.exp(-wedge_soft_beta * delta).to(dt)
        y = torch.fft.ifftn(torch.fft.fftn(y) * M).real.to(dt)

    # (4) noise
    if add_poisson:
        g = float(torch.empty((), device=device).uniform_(*poisson_gain_range).item())
        y = torch.poisson(y.clamp(0,1) * g) / g
    if add_readnoise:
        sigma_r = float(torch.empty((), device=device).uniform_(*read_noise_std_range).item())
        y = y + torch.randn_like(y) * sigma_r
    if clamp01:
        y.clamp_(0,1)
    return y

class LorentzLaminoBlurTransform:
    """Depth-variant Lorentzian z-blur + optional XY blur."""
    def __init__(self,
                 device="cpu",
                 xy_mode="none", fwhm_xy=0.0, fwhmL_xy=0.0,
                 radius_mult_xy=3.0, tail_mult_xy=15,
                 fwhmL_z_min=2.0, fwhmL_z_max=16.0, p=1.6, K=12,
                 tail_mult_z=20, cap_frac=0.5,
                 include_wedge=False, wedge_alpha_deg=35.0, wedge_soft_beta=2.0,
                 add_noise=False, clamp01=True):
        self.device = torch.device(device)
        self.kw = dict(
            xy_mode=xy_mode, fwhm_xy=fwhm_xy, fwhmL_xy=fwhmL_xy,
            radius_mult_xy=radius_mult_xy, tail_mult_xy=tail_mult_xy,
            fwhmL_z_min=fwhmL_z_min, fwhmL_z_max=fwhmL_z_max,
            p=p, K=K, tail_mult_z=tail_mult_z, cap_frac=cap_frac,
            wedge_alpha_deg=(wedge_alpha_deg if include_wedge else None),
            wedge_soft_beta=wedge_soft_beta,
            add_poisson=add_noise, add_readnoise=add_noise, clamp01=clamp01,
        )

    def __call__(self, vol_cpu_float01: torch.Tensor) -> torch.Tensor:
        v = vol_cpu_float01.to(self.device, non_blocking=True)
        y = lamino_blur_lorentz(v, **self.kw)
        return y.to("cpu", non_blocking=True)
