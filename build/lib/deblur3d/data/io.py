# src/deblur3d/data/io.py
import numpy as np, tifffile as tiff

def read_volume_float01(path):
    vol = tiff.imread(path)
    if vol.ndim == 2:
        vol = vol[None, ...]
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D or 2D tif, got {vol.shape}")
    if np.issubdtype(vol.dtype, np.integer):
        vol = vol.astype(np.float32) / max(np.iinfo(vol.dtype).max, 1)
    else:
        vol = vol.astype(np.float32)
        vmin, vmax = float(vol.min()), float(vol.max())
        if vmin < 0 or vmax > 1.5:
            lo, hi = np.percentile(vol, [1, 99.9])
            vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
        else:
            vol = np.clip(vol, 0, 1)
    return vol
