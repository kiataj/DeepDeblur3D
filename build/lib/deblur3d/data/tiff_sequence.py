# src/deblur3d/data/tiff_sequence.py
import os, glob, math, random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import tifffile as tiff
except Exception as e:
    raise RuntimeError("tifffile is required for TiffDataset. pip install tifffile") from e

__all__ = ["TiffDataset"]

def _natural_key(name: str):
    import re
    num_re = re.compile(r'(\d+)')
    parts = num_re.split(name)
    parts[1::2] = [int(p) for p in parts[1::2]]
    return parts

def _list_slices(folder: Path) -> List[Path]:
    exts = (".tif", ".tiff")
    files = [p for p in folder.iterdir()
             if p.is_file()
             and p.suffix.lower() in exts
             and not p.name.lower().startswith("proj_")
             and not p.name.lower().startswith("mask_")]
    files.sort(key=lambda p: _natural_key(p.name))
    return files

@dataclass
class VolumeInfo:
    root_dir: Path
    n_slices: int
    H: int
    W: int
    lo: float
    hi: float
    spacing_y: Optional[float] = None
    spacing_x: Optional[float] = None

class TiffDataset(Dataset):
    """
    Stream random 3D patches from TIFF slice sequences listed in a manifest (parquet/csv).
    Returns (sharp, blurred) as torch.float32 tensors in [0,1] shaped (D,H,W).

    Parameters
    ----------
    manifest_path : str or Path
        Path to manifest parquet/csv created by scripts/index_from_excel.py
    split : Optional[str]
        If your manifest has a 'split' column ('train'/'val'/'test'), filter by it.
    patch_size : Tuple[int,int,int]
        (D,H,W) patch shape in voxels.
    blur_transform : Optional[callable]
        A CPU transform callable: sharp -> blurred (torch.Tensor -> torch.Tensor). If None, returns sharp twice.
    balance : str
        'volume' (uniform among volumes) or 'slice_count' (weight by n_slices).
    samples_per_epoch : Optional[int]
        Length reported by __len__. If None, defaults to number of volumes.
    """
    def __init__(self,
                 manifest_path: str | Path,
                 split: Optional[str] = None,
                 patch_size: Tuple[int,int,int] = (96,128,128),
                 blur_transform=None,
                 balance: str = "volume",
                 samples_per_epoch: Optional[int] = None):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.blur_transform = blur_transform
        self.balance = balance
        self.samples_per_epoch = samples_per_epoch

        mpath = Path(manifest_path)
        if not mpath.exists():
            raise FileNotFoundError(f"Manifest not found: {mpath}")

        if mpath.suffix.lower() == ".parquet":
            df = pd.read_parquet(mpath)
        else:
            df = pd.read_csv(mpath)

        if split is not None and "split" in df.columns:
            df = df[df["split"].astype(str).str.lower() == str(split).lower()].copy()

        if df.empty:
            raise RuntimeError("Manifest has no rows after filtering.")

        self.df = df.reset_index(drop=True)
        self.vols: List[VolumeInfo] = []
        for _, r in self.df.iterrows():
            root = Path(r["root_dir"])
            if not root.exists():
                # skip missing dirs; keep consistency by warning but not crashing
                continue
            self.vols.append(VolumeInfo(
                root_dir=root,
                n_slices=int(r["n_slices"]),
                H=int(r["H"]), W=int(r["W"]),
                lo=float(r.get("lo", 0.0)),
                hi=float(r.get("hi", 1.0)),
                spacing_y=None if pd.isna(r.get("spacing_y", np.nan)) else float(r["spacing_y"]),
                spacing_x=None if pd.isna(r.get("spacing_x", np.nan)) else float(r["spacing_x"]),
            ))
        if not self.vols:
            raise RuntimeError("No valid volumes found on disk.")

        # build sampling weights
        if self.balance == "slice_count":
            arr = np.array([max(1, v.n_slices) for v in self.vols], dtype=np.float64)
            self.weights = (arr / arr.sum()).tolist()
        else:
            self.weights = None  # uniform

        # lazy slice lists (per volume)
        self._slices_cache: Dict[int, List[Path]] = {}

    def __len__(self) -> int:
        return self.samples_per_epoch if self.samples_per_epoch is not None else len(self.vols)

    def _pick_volume_index(self) -> int:
        if self.weights is None:
            return random.randrange(len(self.vols))
        return int(np.random.choice(len(self.vols), p=self.weights))

    def _get_slice_paths(self, vidx: int) -> List[Path]:
        if vidx not in self._slices_cache:
            paths = _list_slices(self.vols[vidx].root_dir)
            if len(paths) == 0:
                raise RuntimeError(f"No slices in {self.vols[vidx].root_dir}")
            self._slices_cache[vidx] = paths
        return self._slices_cache[vidx]

    def _read_stack(self, paths: List[Path], z0: int, D: int) -> torch.Tensor:
        # Read [z0 : z0+D) and stack into (D,H,W) float32
        zs = range(z0, min(z0 + D, len(paths)))
        imgs = []
        for zi in zs:
            arr = tiff.imread(str(paths[zi]))
            if arr.ndim != 2:
                raise RuntimeError(f"Non-2D slice at {paths[zi]} shape={arr.shape}")
            imgs.append(torch.from_numpy(arr.astype(np.float32)))
        if len(imgs) < D:
            # reflect-pad along z if we hit the end
            need = D - len(imgs)
            reflect = imgs[-2::-1] if len(imgs) > 1 else [imgs[-1]] * need
            imgs.extend(reflect[:need])
        stack = torch.stack(imgs, dim=0)  # (D,H,W)
        stack = stack.contiguous().clone()
        return stack

    @staticmethod
    def _norm01(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        y = (x - lo) / max(hi - lo, 1e-6)
        return y.clamp_(0.0, 1.0)

    @torch.no_grad()
    def __getitem__(self, idx: int):
        # choose a volume (randomized each call for streaming behavior)
        vidx = self._pick_volume_index()
        v = self.vols[vidx]
        D, H, W = self.patch_size

        # fetch slice list and sanity-check dims
        slices = self._get_slice_paths(vidx)
        H0, W0 = v.H, v.W

        # choose spatial crop (reflect-pad to ensure valid region)
        def _rand_start(full, size):
            return 0 if full <= size else random.randint(0, full - size)
        z0 = _rand_start(len(slices), D)
        y0 = _rand_start(H0, H)
        x0 = _rand_start(W0, W)

        # read and crop
        vol = self._read_stack(slices, z0=z0, D=D)  # (D,H0,W0)
        if H0 < H or W0 < W:
            pad_h = max(0, H - H0)
            pad_w = max(0, W - W0)
            vol = F.pad(vol.unsqueeze(0).unsqueeze(0),
                        (pad_w//2, pad_w - pad_w//2,
                         pad_h//2, pad_h - pad_h//2,
                         0, 0), mode="reflect").squeeze(0).squeeze(0)
            H0, W0 = vol.shape[1], vol.shape[2]
            y0 = _rand_start(H0, H)
            x0 = _rand_start(W0, W)

        patch = vol[:, y0:y0+H, x0:x0+W].contiguous()  # (D,H,W)

        # normalize per manifest lo/hi
        sharp = self._norm01(patch, v.lo, v.hi)

        # blurred variant
        if self.blur_transform is None:
            blurred = sharp.clone()
        else:
            blurred = self.blur_transform(sharp)
            
        # ensure PyTorch-owned, resizable storages (not NumPy-backed)
        sharp   = sharp.contiguous().clone()
        blurred = blurred.contiguous().clone()

        return sharp, blurred
