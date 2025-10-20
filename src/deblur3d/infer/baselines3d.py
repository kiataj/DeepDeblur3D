#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU 3D baseline filters for DeepDeblur3D with selectable runs.

Baselines:
- USM (unsharp masking)
- LoG sharpen
- Wiener deconvolution (Gaussian OTF)
- Richardson–Lucy deconvolution (Gaussian PSF)

Utilities:
- Gaussian kernel / separable 3D blur
- 3D Laplacian
- Timing wrapper
"""

from __future__ import annotations
import time
from typing import Dict, Tuple, Optional, Iterable

import numpy as np
import torch
import torch.nn.functional as F

# -------------------------- dtype/format helpers --------------------------

def as_torch3d01(x: np.ndarray, device: torch.device | str = "cuda") -> torch.Tensor:
    """np.float32 (D,H,W) in [0,1] -> torch.float32 (D,H,W) on device."""
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array (D,H,W), got shape={x.shape}")
    return torch.from_numpy(x.astype(np.float32, copy=False)).to(device)

def to_numpy01(x: torch.Tensor) -> np.ndarray:
    """torch.Tensor (D,H,W) in [0,1] -> np.float32 (D,H,W)."""
    return x.detach().clamp(0, 1).to("cpu").numpy().astype(np.float32)

# -------------------------- Gaussian kernels/convs --------------------------

@torch.no_grad()
def _gauss1d(sigma: float, device, dtype=torch.float32, radius_mult: float = 3.0) -> Tuple[torch.Tensor, int]:
    import math
    sigma = max(1e-6, float(sigma))
    r = max(1, int(math.ceil(radius_mult * sigma)))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / (k.sum() + 1e-12)
    return k, r

@torch.no_grad()
def gaussian_blur3d_tensor(x: torch.Tensor, sigma: float, pad_mode: str = "reflect") -> torch.Tensor:
    """Separable 3D Gaussian blur on (B,C,D,H,W) tensor, returns same shape."""
    if sigma <= 0:
        return x
    if x.ndim != 5:
        raise ValueError(f"Expected 5D tensor (B,C,D,H,W), got {tuple(x.shape)}")
    k1d, r = _gauss1d(sigma, x.device, x.dtype)
    kz = k1d.view(1, 1, -1, 1, 1)
    ky = k1d.view(1, 1, 1, -1, 1)
    kx = k1d.view(1, 1, 1, 1, -1)
    C = x.shape[1]
    y = F.conv3d(F.pad(x, (0, 0, 0, 0, r, r), mode=pad_mode), kz, groups=C)
    y = F.conv3d(F.pad(y, (0, 0, r, r, 0, 0), mode=pad_mode), ky, groups=C)
    y = F.conv3d(F.pad(y, (r, r, 0, 0, 0, 0), mode=pad_mode), kx, groups=C)
    return y

@torch.no_grad()
def laplacian3d_tensor(x: torch.Tensor, pad_mode: str = "reflect") -> torch.Tensor:
    """3D 6-neighborhood Laplacian on (B,C,D,H,W)."""
    if x.ndim != 5:
        raise ValueError(f"Expected 5D tensor (B,C,D,H,W), got {tuple(x.shape)}")
    w = torch.zeros((1, 1, 3, 3, 3), device=x.device, dtype=x.dtype)
    w[0, 0, 1, 1, 1] = 6.0
    w[0, 0, 1, 1, 0] = w[0, 0, 1, 1, 2] = -1.0
    w[0, 0, 1, 0, 1] = w[0, 0, 1, 2, 1] = -1.0
    w[0, 0, 0, 1, 1] = w[0, 0, 2, 1, 1] = -1.0
    xpad = F.pad(x, (1, 1, 1, 1, 1, 1), mode=pad_mode)
    return F.conv3d(xpad, w)

# -------------------------- Baseline filters (GPU) --------------------------

@torch.no_grad()
def usm3d_gpu(vol: np.ndarray | torch.Tensor, sigma: float, amount: float,
              pad_mode: str = "reflect", device: torch.device | str = "cuda") -> np.ndarray:
    """Unsharp masking: y = clamp(x + amount * (x - G_sigma(x)), 0, 1)"""
    x = as_torch3d01(vol, device) if isinstance(vol, np.ndarray) else vol.to(device)
    x4 = x.view(1, 1, *x.shape)
    base = gaussian_blur3d_tensor(x4, sigma, pad_mode=pad_mode)
    y = (x4 + amount * (x4 - base)).clamp(0, 1)
    return to_numpy01(y.view(*x.shape))

@torch.no_grad()
def log_sharpen3d_gpu(vol: np.ndarray | torch.Tensor, sigma: float, lam: float,
                      pad_mode: str = "reflect", device: torch.device | str = "cuda") -> np.ndarray:
    """LoG sharpen: y = clamp(x - λ * Laplace(G_sigma(x)), 0, 1)"""
    x = as_torch3d01(vol, device) if isinstance(vol, np.ndarray) else vol.to(device)
    x4 = x.view(1, 1, *x.shape)
    g  = gaussian_blur3d_tensor(x4, sigma, pad_mode=pad_mode)
    L  = laplacian3d_tensor(g, pad_mode=pad_mode)
    y  = (x4 - lam * L).clamp(0, 1)
    return to_numpy01(y.view(*x.shape))

@torch.no_grad()
def wiener_gaussian3d_gpu(vol: np.ndarray | torch.Tensor, sigma: float, K: float = 0.01,
                          device: torch.device | str = "cuda") -> np.ndarray:
    """Wiener deconvolution with Gaussian OTF (frequency-domain)."""
    x = as_torch3d01(vol, device) if isinstance(vol, np.ndarray) else vol.to(device)
    D, H, W = x.shape
    X = torch.fft.fftn(x)
    fz = torch.fft.fftfreq(D, d=1.0, device=x.device).view(D, 1, 1)
    fy = torch.fft.fftfreq(H, d=1.0, device=x.device).view(1, H, 1)
    fx = torch.fft.fftfreq(W, d=1.0, device=x.device).view(1, 1, W)
    two_pi2 = (2.0 * np.pi) ** 2
    Htf = torch.exp(-0.5 * two_pi2 * (sigma ** 2) * (fz * fz + fy * fy + fx * fx))
    Y = X * Htf / (Htf * Htf + K)
    y = torch.fft.ifftn(Y).real.clamp(0, 1)
    return to_numpy01(y)

@torch.no_grad()
def richardson_lucy3d_gpu(vol: np.ndarray | torch.Tensor, sigma: float, n_iter: int = 15,
                           device: torch.device | str = "cuda") -> np.ndarray:
    """Richardson–Lucy deconvolution with Gaussian PSF, multiplicative updates."""
    x = as_torch3d01(vol, device) if isinstance(vol, np.ndarray) else vol.to(device)
    x4 = x.view(1, 1, *x.shape).clamp_min(1e-6)
    k1d, r = _gauss1d(sigma, x4.device, x4.dtype)
    kz = k1d.view(1, 1, -1, 1, 1)
    ky = k1d.view(1, 1, 1, -1, 1)
    kx = k1d.view(1, 1, 1, 1, -1)

    def psfZ(z): return F.conv3d(F.pad(z, (0, 0, 0, 0, r, r), mode="replicate"), kz)
    def psfY(z): return F.conv3d(F.pad(z, (0, 0, r, r, 0, 0), mode="replicate"), ky)
    def psfX(z): return F.conv3d(F.pad(z, (r, r, 0, 0, 0, 0), mode="replicate"), kx)
    def psf_conv(z):  return psfX(psfY(psfZ(z)))

    y = x4.clone()
    for _ in range(int(n_iter)):
        est   = psf_conv(y).clamp_min(1e-6)
        ratio = x4 / est
        y     = (y * psf_conv(ratio)).clamp(0, 1)

    return to_numpy01(y.view(*x.shape))

# -------------------------- Timing helper --------------------------

def run_timed_cuda(name: str, fn, *args, **kwargs) -> Tuple[np.ndarray, float]:
    """Run CUDA op with accurate timing (synchronize before/after)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0)

# -------------------------- Selection wrapper --------------------------

AVAILABLE: Dict[str, str] = {
    "USM": "Unsharp masking",
    "LoG": "Laplacian-of-Gaussian sharpen",
    "Wiener": "Wiener deconvolution (Gaussian OTF)",
    "RL": "Richardson–Lucy deconvolution (Gaussian PSF)",
}

def run_baselines(
    vol: np.ndarray,
    *,
    run: Iterable[str] = ("USM", "LoG", "Wiener", "RL"),
    fwhm_vox: float = 9.0,
    usm_amount: float = 2.0,
    log_lambda: float = 2.0,
    wiener_K: float = 0.015,
    rl_iters: int = 10,
    device: torch.device | str = "cuda",
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Execute a selectable subset of baselines on a [0,1] np.float32 volume.
      run: names to run (subset of AVAILABLE keys).
    Returns:
      results[name] -> np.ndarray (D,H,W)
      times[name]   -> float seconds
    """
    if vol.ndim != 3:
        raise ValueError(f"Expected (D,H,W), got {vol.shape}")
    run = tuple(run)
    unknown = [r for r in run if r not in AVAILABLE]
    if unknown:
        raise ValueError(f"Unknown baseline(s): {unknown}. Available: {list(AVAILABLE)}")
    sigma = float(fwhm_vox) / 2.3548

    results: Dict[str, np.ndarray] = {}
    times:   Dict[str, float] = {}

    if "USM" in run:
        out, dt = run_timed_cuda("USM", usm3d_gpu, vol, sigma, usm_amount, device=device)
        results["USM"] = out; times["USM"] = dt

    if "LoG" in run:
        out, dt = run_timed_cuda("LoG", log_sharpen3d_gpu, vol, sigma, log_lambda, device=device)
        results["LoG"] = out; times["LoG"] = dt

    if "Wiener" in run:
        out, dt = run_timed_cuda("Wiener", wiener_gaussian3d_gpu, vol, sigma, wiener_K, device=device)
        results["Wiener"] = out; times["Wiener"] = dt

    if "RL" in run:
        out, dt = run_timed_cuda("RL", richardson_lucy3d_gpu, vol, sigma, rl_iters, device=device)
        results["RL"] = out; times["RL"] = dt

    return results, times

# -------------------------- Optional: CLI --------------------------

def _try_read_tif(path: str) -> np.ndarray:
    try:
        import tifffile as tiff
    except Exception as e:
        raise RuntimeError("Reading TIFF requires 'tifffile'. pip install tifffile") from e
    vol = tiff.imread(path).astype(np.float32)
    # robust scale to [0,1]
    lo, hi = np.percentile(vol, (1.0, 99.5))
    if hi <= lo: hi = lo + 1e-6
    vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    return vol

if __name__ == "__main__":
    import argparse, os, json
    ap = argparse.ArgumentParser(description="Run selectable 3D baseline filters on a TIFF")
    ap.add_argument("--vol", required=True, help="Path to input .tif (3D multi-page)")
    ap.add_argument("--run", action="append", choices=list(AVAILABLE), help="Baseline to run (repeatable)", required=True)
    ap.add_argument("--fwhm", type=float, default=9.0, help="FWHM in voxels (Gaussian PSF)")
    ap.add_argument("--usm-amount", type=float, default=2.0, help="USM amount")
    ap.add_argument("--log-lambda", type=float, default=2.0, help="LoG sharpening lambda")
    ap.add_argument("--wiener-K", type=float, default=0.015, help="Wiener noise param K")
    ap.add_argument("--rl-iters", type=int, default=10, help="Richardson–Lucy iterations")
    ap.add_argument("--outdir", default="baselines_out", help="Folder to write NPY results")
    ap.add_argument("--save-json", default=None, help="Optional path to save timings as JSON")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vol = _try_read_tif(args.vol)
    results, times = run_baselines(
        vol,
        run=args.run,
        fwhm_vox=args.fwhm,
        usm_amount=args.usm_amount,
        log_lambda=args.log_lambda,
        wiener_K=args.wiener_K,
        rl_iters=args.rl_iters,
        device=dev,
    )
    os.makedirs(args.outdir, exist_ok=True)
    for k, v in results.items():
        np.save(os.path.join(args.outdir, f"{k}.npy"), v)
    print("Times (s):", {k: f"{t:.3f}" for k, t in times.items()})
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(times, f, indent=2)
