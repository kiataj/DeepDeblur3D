#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
No-reference 3D quality metrics for volumetric data (float32 in [0,1]).

Includes:
- Tenengrad (3D gradient energy)
- Variance of 3D Laplacian
- High-frequency energy ratio via FFT (r > r0 of Nyquist)
- Input-anchored noise MAD on a fixed 'flat' mask
- Noise Reduction Factor (NRF) relative to input
- Automatic CNR (aCNR) using input-based partition (masked Otsu) and robust sigmas

All heavy ops run on GPU if available; otherwise fall back to CPU.
"""

from __future__ import annotations
from typing import Dict, Tuple, Iterable, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


__all__ = [
    "center_crop_np",
    "tenengrad_3d",
    "lap_var_3d",
    "gauss_separable",
    "build_flat_mask_from_input",
    "noise_mad_hp_masked",
    "hf_energy_ratio",
    "evaluate_methods_no_gt",
    "add_auto_cnr_columns",
]


# ------------------------- device / dtype helpers -------------------------

def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _to_t(x: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
    """np.float32 (D,H,W) → torch.float32 (D,H,W) on device."""
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array (D,H,W), got {x.shape}")
    dev = _default_device() if device is None else device
    return torch.from_numpy(x.astype(np.float32, copy=False)).to(device=dev, dtype=torch.float32)

def center_crop_np(x: np.ndarray, maxD=128, maxH=256, maxW=256) -> np.ndarray:
    """Center-crop a 3D numpy array to at most (maxD,maxH,maxW)."""
    D, H, W = x.shape
    d = min(D, maxD); h = min(H, maxH); w = min(W, maxW)
    zs = (D - d) // 2; ys = (H - h) // 2; xs = (W - w) // 2
    return x[zs:zs+d, ys:ys+h, xs:xs+w].copy()


# ------------------------- core 3D filters / metrics -------------------------

@torch.no_grad()
def tenengrad_3d(x: torch.Tensor) -> float:
    """Tenengrad: mean squared 3D gradient magnitude on (D,H,W) tensor."""
    if x.ndim != 3:
        raise ValueError(f"Expected (D,H,W), got {tuple(x.shape)}")
    # simple 3D Sobel-like derivative (-1,0,1)
    d = torch.tensor([1, 0, -1], dtype=x.dtype, device=x.device)
    kz = d.view(1, 1, 3, 1, 1); ky = d.view(1, 1, 1, 3, 1); kx = d.view(1, 1, 1, 1, 3)
    x4 = x.unsqueeze(0).unsqueeze(0)
    gz = F.conv3d(F.pad(x4, (0,0,0,0,1,1), "replicate"), kz)
    gy = F.conv3d(F.pad(x4, (0,0,1,1,0,0), "replicate"), ky)
    gx = F.conv3d(F.pad(x4, (1,1,0,0,0,0), "replicate"), kx)
    return float((gx*gx + gy*gy + gz*gz).mean().item())

@torch.no_grad()
def lap_var_3d(x: torch.Tensor) -> float:
    """Variance of 3D 6-neighborhood Laplacian on (D,H,W) tensor."""
    if x.ndim != 3:
        raise ValueError(f"Expected (D,H,W), got {tuple(x.shape)}")
    w = torch.zeros((1, 1, 3, 3, 3), device=x.device, dtype=x.dtype)
    w[0,0,1,1,1] = 6.0
    w[0,0,1,1,0] = w[0,0,1,1,2] = -1.0
    w[0,0,1,0,1] = w[0,0,1,2,1] = -1.0
    w[0,0,0,1,1] = w[0,0,2,1,1] = -1.0
    y = F.conv3d(F.pad(x.unsqueeze(0).unsqueeze(0), (1,1,1,1,1,1), "replicate"), w).squeeze()
    return float(y.var().item())

@torch.no_grad()
def gauss_separable(x4: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Separable 3D Gaussian blur for (1,1,D,H,W) tensor.
    Returns same shape; reflect pad; kernel radius = ceil(3*sigma).
    """
    import math
    s = max(1e-6, float(sigma))
    r = max(1, int(math.ceil(3 * s)))
    z = torch.arange(-r, r + 1, device=x4.device, dtype=x4.dtype)
    g = torch.exp(-(z * z) / (2 * s * s)); g = g / (g.sum() + 1e-12)
    kz = g.view(1, 1, -1, 1, 1); ky = g.view(1, 1, 1, -1, 1); kx = g.view(1, 1, 1, 1, -1)
    y = F.conv3d(F.pad(x4, (0,0,0,0,r,r), "reflect"), kz)
    y = F.conv3d(F.pad(y,  (0,0,r,r,0,0), "reflect"), ky)
    y = F.conv3d(F.pad(y,  (r,r,0,0,0,0), "reflect"), kx)
    return y

@torch.no_grad()
def build_flat_mask_from_input(
    x_in: torch.Tensor,
    flat_pct: float = 0.30,
    min_vox: int = 32_768
) -> torch.Tensor:
    """
    Build a low-texture (flat) boolean mask from the INPUT volume.
    - Pre-blur input (σ≈0.7) to reduce edge influence
    - Use gradient magnitude quantile to select 'flat_pct' fraction
    - Guarantee at least `min_vox` voxels by fallback selection
    """
    if x_in.ndim != 3:
        raise ValueError(f"Expected (D,H,W), got {tuple(x_in.shape)}")

    t = x_in.unsqueeze(0).unsqueeze(0)
    t_s = gauss_separable(t, sigma=0.7)
    d = torch.tensor([1, 0, -1], dtype=t.dtype, device=t.device)
    kz = d.view(1, 1, 3, 1, 1); ky = d.view(1, 1, 1, 3, 1); kx = d.view(1, 1, 1, 1, 3)
    gx = F.conv3d(F.pad(t_s, (0,0,0,0,1,1), "replicate"), kz).abs()
    gy = F.conv3d(F.pad(t_s, (0,0,1,1,0,0), "replicate"), ky).abs()
    gz = F.conv3d(F.pad(t_s, (1,1,0,0,0,0), "replicate"), kx).abs()
    grad = (gx + gy + gz).squeeze()

    # quantile threshold on gradient magnitude
    q = torch.quantile(grad, torch.tensor(flat_pct, device=grad.device))
    m = (grad <= q)

    # ensure enough voxels
    if m.sum().item() < min_vox:
        ksel = min(min_vox, grad.numel())
        _, idx = torch.topk((-grad).flatten(), k=ksel)  # smallest gradients
        m = torch.zeros_like(grad, dtype=torch.bool).flatten()
        m[idx] = True
        m = m.view_as(grad)
    return m  # (D,H,W) bool

@torch.no_grad()
def noise_mad_hp_masked(x: torch.Tensor, mask: torch.Tensor, sigma: float = 1.0) -> float:
    """
    MAD of high-pass residual over fixed mask.
    - low = G_sigma(x)
    - hp  = x - low
    - MAD(hp[mask]) * 1.4826 (Gaussian equiv)
    """
    if x.ndim != 3 or mask.ndim != 3:
        raise ValueError("x and mask must be (D,H,W)")
    x4 = x.unsqueeze(0).unsqueeze(0)
    low = gauss_separable(x4, sigma=sigma).squeeze()
    hp = (x - low)
    r = hp[mask]
    if r.numel() == 0:
        return float("nan")
    med = r.median()
    mad = (r - med).abs().median() * 1.4826
    return float(mad.item() + 1e-12)

@torch.no_grad()
def hf_energy_ratio(x: torch.Tensor, r0: float = 0.6) -> float:
    """
    Ratio of spectral energy at normalized radius r > r0 (0..1 Nyquist) to total energy.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected (D,H,W), got {tuple(x.shape)}")
    D, H, W = x.shape
    X = torch.fft.fftn(x)
    P = (X.abs() ** 2)
    fz = torch.fft.fftfreq(D, d=1.0, device=x.device).view(D, 1, 1)
    fy = torch.fft.fftfreq(H, d=1.0, device=x.device).view(1, H, 1)
    fx = torch.fft.fftfreq(W, d=1.0, device=x.device).view(1, 1, W)
    fny = 0.5
    r = torch.sqrt((fz/fny)**2 + (fy/fny)**2 + (fx/fny)**2)
    mask = (r >= r0)
    return float(P[mask].sum() / (P.sum() + 1e-12))

# ------------------------- evaluation wrappers -------------------------

def evaluate_methods_no_gt(
    outputs: Dict[str, np.ndarray],
    vol_input: np.ndarray,
    *,
    crop: Tuple[int, int, int] = (128, 256, 256),
    hp_sigma_noise: float = 1.0,
    flat_pct: float = 0.30,
    min_vox: int = 32_768,
    hf_r0: float = 0.6,
    device: Optional[torch.device] = None,
    times: Optional[Dict[str, float]] = None,   # optional runtime dict to join
) -> pd.DataFrame:
    """
    Compute no-reference metrics for multiple methods, using a single
    input-anchored 'flat' mask (low texture) shared across methods.

    outputs: mapping name -> np.ndarray (D,H,W) in [0,1]; should include "Input"
    vol_input: original input volume (used to build mask and NRF reference)
    crop: center crop size for metrics (reduces FFT memory)
    times: optional dict {method: seconds} to merge as 'seconds' column
    """
    dev = _default_device() if device is None else device

    # fixed input crop & mask
    xin_c = center_crop_np(vol_input, *crop)
    x_in = _to_t(xin_c, device=dev)
    flat_mask = build_flat_mask_from_input(x_in, flat_pct=flat_pct, min_vox=min_vox)
    noise_in = noise_mad_hp_masked(x_in, flat_mask, sigma=hp_sigma_noise)

    rows = []
    hf_col = f"HF_ratio@r>{hf_r0}"
    for name, arr in outputs.items():
        arr_c = center_crop_np(arr, *crop)
        y = _to_t(arr_c, device=dev)

        row = {"method": name}
        row["Tenengrad"] = tenengrad_3d(y)
        row["Var(Lap)"]  = lap_var_3d(y)
        row[hf_col]      = hf_energy_ratio(y, r0=hf_r0)

        noise = noise_mad_hp_masked(y, flat_mask, sigma=hp_sigma_noise)
        row["Noise_MAD"] = noise
        row["NRF"]       = float(noise / (noise_in + 1e-12))  # <1 ⇒ denoised vs input
        rows.append(row)

    df = pd.DataFrame(rows).set_index("method")

    # join runtimes (if provided)
    if isinstance(times, dict):
        t = pd.Series(times, name="seconds")
        df = df.join(t, how="left")

    # column order
    cols = [c for c in ["Tenengrad", "Var(Lap)", hf_col, "Noise_MAD", "NRF", "seconds"] if c in df.columns]
    df = df[cols]

    # sort: sharpness↑ then NRF↓ (if present)
    sort_keys, asc = [], []
    if "Tenengrad" in df.columns: sort_keys.append("Tenengrad"); asc.append(False)
    if "NRF" in df.columns:       sort_keys.append("NRF");       asc.append(True)
    if sort_keys:
        df = df.sort_values(by=sort_keys, ascending=asc)

    return df


# ------------------------- automatic CNR (no manual ROIs) -------------------------

@torch.no_grad()
def _otsu_threshold_from_masked(x: torch.Tensor, mask: torch.Tensor, bins: int = 256) -> float:
    """Otsu threshold computed on x[mask] ∈ [0,1]."""
    vals = x[mask].clamp(0, 1).detach().float().cpu().numpy()
    if vals.size < 1024:  # too few voxels → fallback to mean
        return float(vals.mean()) if vals.size > 0 else 0.5
    hist, edges = np.histogram(vals, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    w = hist.sum()
    if w <= 0:
        return 0.5
    p = hist / w
    omega = np.cumsum(p)
    centers = (edges[:-1] + edges[1:]) * 0.5
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    k = int(np.nanargmax(sigma_b))
    return float((edges[k] + edges[k + 1]) * 0.5)

@torch.no_grad()
def _robust_sigma(v: torch.Tensor) -> torch.Tensor:
    """Robust σ via MAD (Gaussian equiv). v is 1D tensor."""
    if v.numel() == 0:
        return torch.tensor(float("nan"), device=v.device)
    med = v.median()
    mad = (v - med).abs().median()
    return mad * 1.4826

@torch.no_grad()
def _auto_cnr_on_fixed_partition(
    out_y: torch.Tensor,
    x_in: torch.Tensor,
    flat_mask: torch.Tensor,
    thr: float,
    robust: bool = True,
    min_class_vox: int = 4096,
) -> Tuple[float, float, float, float, float, int, int]:
    """
    CNR on two classes defined ON INPUT (x_in <= thr vs > thr) within flat_mask.
    Returns: (cnr, mu0, mu1, sig0, sig1, n0, n1)
    """
    m0 = flat_mask & (x_in <= thr)
    m1 = flat_mask & (x_in >  thr)

    # ensure both classes have voxels; fallback to quantiles
    if m0.sum().item() < min_class_vox or m1.sum().item() < min_class_vox:
        q0, q1 = torch.quantile(x_in[flat_mask], torch.tensor([0.3, 0.7], device=x_in.device))
        m0 = flat_mask & (x_in <= q0)
        m1 = flat_mask & (x_in >= q1)

    v0 = out_y[m0]; v1 = out_y[m1]
    s0 = _robust_sigma(v0) if robust else v0.std(unbiased=False)
    s1 = _robust_sigma(v1) if robust else v1.std(unbiased=False)
    mu0 = v0.mean(); mu1 = v1.mean()
    cnr = (mu1 - mu0).abs() / torch.sqrt(s0 * s0 + s1 * s1 + 1e-12)
    return (float(cnr.item()), float(mu0.item()), float(mu1.item()),
            float(s0.item()),  float(s1.item()),  int(v0.numel()),  int(v1.numel()))

def add_auto_cnr_columns(
    df: pd.DataFrame,
    outputs: Dict[str, np.ndarray],
    vol_input: np.ndarray,
    vx_size: float,
    *,
    crop: Tuple[int, int, int] = (128, 256, 256),
    hp_sigma_noise: float = 1.0,
    flat_pct: float = 0.30,
    min_vox: int = 32_768,
    robust: bool = True,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Append automatic CNR columns:
      - aCNR computed using input-defined classes (masked Otsu, within flat mask)
      - aCNR/(2.4*vx) normalized by voxel size (domain heuristic)
    """
    dev = _default_device() if device is None else device

    xin_c = center_crop_np(vol_input, *crop)
    x_in = _to_t(xin_c, device=dev)

    flat_mask = build_flat_mask_from_input(x_in, flat_pct=flat_pct, min_vox=min_vox)
    thr = _otsu_threshold_from_masked(x_in, flat_mask, bins=256)

    acnr, acnr_norm = {}, {}
    for name, arr in outputs.items():
        arr_c = center_crop_np(arr, *crop)
        y = _to_t(arr_c, device=dev)
        cnr, *_ = _auto_cnr_on_fixed_partition(y, x_in, flat_mask, thr, robust=robust)
        acnr[name] = cnr
        acnr_norm[name] = cnr / (2.4 * float(vx_size) + 1e-12)

    df["aCNR"] = pd.Series(acnr)
    df["aCNR/(2.4*vx)"] = pd.Series(acnr_norm)
    return df
