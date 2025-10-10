# src/deblur3d/data/tiff_volume.py
import os, glob, numpy as np, tifffile as tiff, torch
import torch.nn.functional as F
from torch.utils.data import Dataset

__all__ = ["TiffVolumeDataset"]

def _normalize_to_float01(arr: np.ndarray) -> torch.Tensor:
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iio(arr.dtype) if hasattr(np, "iio") else np.iinfo(arr.dtype)
        return torch.from_numpy(arr.astype(np.float32) / max(info.max, 1))
    return torch.from_numpy(arr.astype(np.float32))

def _random_flip_rot_90(x: torch.Tensor):
    if torch.rand(1) < 0.5: x = x.flip(1)
    if torch.rand(1) < 0.5: x = x.flip(2)
    k = torch.randint(0, 4, (1,)).item()
    if k: x = x.rot90(k, dims=(1,2))
    return x

def _crop_or_pad_to(x: torch.Tensor, target):
    D,H,W = x.shape; td,th,tw = target
    pd0 = max(0, td-D); ph0 = max(0, th-H); pw0 = max(0, tw-W)
    if pd0 or ph0 or pw0:
        x = F.pad(x.unsqueeze(0).unsqueeze(0),
                  (pw0//2, pw0 - pw0//2, ph0//2, ph0 - ph0//2, pd0//2, pd0 - pd0//2),
                  mode="reflect").squeeze(0).squeeze(0)
        D,H,W = x.shape
    z0 = 0 if td==D else torch.randint(0, D-td+1, (1,)).item()
    y0 = 0 if th==H else torch.randint(0, H-th+1, (1,)).item()
    x0 = 0 if tw==W else torch.randint(0, W-tw+1, (1,)).item()
    return x[z0:z0+td, y0:y0+th, x0:x0+tw].contiguous()

class TiffVolumeDataset(Dataset):
    """Reads 3D TIFF volumes, ignores files starting with 'mask_', returns (sharp, blurred)."""
    def __init__(self, root, patch_size=(96,128,128), blur_transform=None, augment=True):
        self.paths = sorted(glob.glob(os.path.join(root, "*.tif")))
        self.paths = [p for p in self.paths if not os.path.basename(p).lower().startswith("mask_")]
        if not self.paths:
            raise RuntimeError("No .tif files (excluding mask_*) found.")
        self.patch_size = patch_size
        self.blur_transform = blur_transform
        self.augment = augment

    def __len__(self): return len(self.paths)

    def _read_tif(self, p):
        vol = tiff.imread(p)
        if vol.ndim == 2: vol = vol[None, ...]
        if vol.ndim != 3: raise RuntimeError(f"Unexpected shape {vol.shape} for {p}")
        return vol

    def __getitem__(self, idx):
        v_np = self._read_tif(self.paths[idx])
        v = _normalize_to_float01(v_np)              # CPU float32
        v = _crop_or_pad_to(v, self.patch_size)
        if self.augment: v = _random_flip_rot_90(v)
        sharp = v.clone()
        blurred = self.blur_transform(sharp) if self.blur_transform else sharp.clone()
        return sharp, blurred
