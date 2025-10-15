# src/deblur3d/data/tiff_volume.py
import os, glob, numpy as np, tifffile as tiff, torch
import torch.nn.functional as F
from torch.utils.data import Dataset

__all__ = ["TiffVolumeDataset", "read_volume_float01"]

def _normalize_to_float01(arr: np.ndarray) -> torch.Tensor:
    """Integer TIFF -> [0,1] float32 tensor; float TIFF -> float32 tensor (no rescale)."""
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return torch.from_numpy(arr.astype(np.float32) / max(info.max, 1))
    return torch.from_numpy(arr.astype(np.float32))

def _crop_or_pad_to(x: torch.Tensor, target):
    D, H, W = x.shape
    td, th, tw = target
    pd0 = max(0, td - D); ph0 = max(0, th - H); pw0 = max(0, tw - W)
    if pd0 or ph0 or pw0:
        x = F.pad(
            x.unsqueeze(0).unsqueeze(0),
            (pw0 // 2, pw0 - pw0 // 2, ph0 // 2, ph0 - ph0 // 2, pd0 // 2, pd0 - pd0 // 2),
            mode="reflect",
        ).squeeze(0).squeeze(0)
        D, H, W = x.shape
    z0 = 0 if td == D else torch.randint(0, D - td + 1, (1,)).item()
    y0 = 0 if th == H else torch.randint(0, H - th + 1, (1,)).item()
    x0 = 0 if tw == W else torch.randint(0, W - tw + 1, (1,)).item()
    return x[z0 : z0 + td, y0 : y0 + th, x0 : x0 + tw].contiguous()

class TiffVolumeDataset(Dataset):
    """Reads 3D TIFF volumes, ignores files starting with 'mask_', returns (sharp, blurred). No augmentations."""
    def __init__(self, root, patch_size=(96, 128, 128), blur_transform=None, augment=False):
        self.paths = sorted(glob.glob(os.path.join(root, "*.tif")))
        self.paths = [p for p in self.paths if not os.path.basename(p).lower().startswith("mask_")]
        if not self.paths:
            raise RuntimeError("No .tif files (excluding mask_*) found.")
        self.patch_size = patch_size
        self.blur_transform = blur_transform
        self.augment = False  # kept for API compatibility; ignored

    def __len__(self): 
        return len(self.paths)

    def _read_tif(self, p):
        vol = tiff.imread(p)
        if vol.ndim == 2: 
            vol = vol[None, ...]
        if vol.ndim != 3: 
            raise RuntimeError(f"Unexpected shape {vol.shape} for {p}")
        return vol

    def __getitem__(self, idx):
        v_np = self._read_tif(self.paths[idx])
        v = _normalize_to_float01(v_np)          # CPU float32 tensor
        v = _crop_or_pad_to(v, self.patch_size)  # random crop (no flips/rotations)
        sharp = v.clone()
        blurred = self.blur_transform(sharp) if self.blur_transform else sharp.clone()
        return sharp, blurred

# --------- public reader for inference/preview (numpy float32 in [0,1]) ----------
def read_volume_float01(path: str) -> np.ndarray:
    """
    Load 2D/3D TIFF and return (D,H,W) float32 in [0,1].
    Integer types are scaled by dtype max; floating types are clipped or
    percentile-normalized to [0,1] if dynamic range is unusual.
    """
    vol = tiff.imread(path)
    if vol.ndim == 2: 
        vol = vol[None, ...]
    if vol.ndim != 3: 
        raise ValueError(f"Expected 3D or 2D tif, got {vol.shape}")

    if np.issubdtype(vol.dtype, np.integer):
        info = np.iinfo(vol.dtype)
        out = vol.astype(np.float32) / max(info.max, 1)
        return np.clip(out, 0, 1)
    else:
        v = vol.astype(np.float32)
        vmin, vmax = float(v.min()), float(v.max())
        if vmin < 0 or vmax > 1.5:
            lo, hi = np.percentile(v, [1, 99.9])
            v = (v - lo) / max(hi - lo, 1e-6)
        return np.clip(v, 0, 1)
