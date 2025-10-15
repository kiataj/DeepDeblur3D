# src/deblur3d/data/dataset.py
import re, random
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

# ---------- helpers ----------

def _natural_key(name: str):
    num_re = re.compile(r"(\d+)")
    parts = num_re.split(name)
    parts[1::2] = [int(p) for p in parts[1::2]]
    return parts

def _list_slices(folder: Path) -> List[Path]:
    exts = (".tif", ".tiff")
    files = [
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in exts
        and not p.name.lower().startswith("proj_")
        and not p.name.lower().startswith("mask_")
    ]
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

# ---------- dataset ----------

class TiffDataset(Dataset):
    """
    Stream random 3D patches from TIFF slice sequences listed in a manifest (parquet/csv).
    Returns (sharp, blurred) as torch.float32 tensors in [0,1] shaped (D,H,W).

    Optional filters/subsetting:
      - filter_query: pandas query (e.g., "n_slices>=256 and size_8bit_GB<3")
      - include_regex / exclude_regex: regex on root_dir
      - min_slices, max_slices, max_size_8bit_GB
      - xml_kV_range=(lo,hi), vox_xy_range_mm=(lo,hi) if xml_* present
      - head_k_per_group + group_col: cap scans per project
      - random_subset: int count or float fraction
      - samples_per_epoch: cap __len__ for faster epochs
      - balance: 'volume' or 'slice_count'
    """
    def __init__(
        self,
        manifest_path: str | Path,
        split: Optional[str] = None,
        patch_size: Tuple[int,int,int] = (96,128,128),
        blur_transform=None,
        balance: str = "volume",
        samples_per_epoch: Optional[int] = None,
        # filters/subsetting
        filter_query: Optional[str] = None,
        include_regex: Optional[str] = None,
        exclude_regex: Optional[str] = None,
        min_slices: Optional[int] = None,
        max_slices: Optional[int] = None,
        max_size_8bit_GB: Optional[float] = None,
        xml_kV_range: Optional[Tuple[float,float]] = None,
        vox_xy_range_mm: Optional[Tuple[float,float]] = None,
        head_k_per_group: Optional[int] = None,
        group_col: Optional[str] = "project_path",
        random_subset: Optional[int | float] = None,
        seed: int = 0,
    ):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.blur_transform = blur_transform
        self.balance = balance
        self.samples_per_epoch = samples_per_epoch
        self._rng = random.Random(seed)

        mpath = Path(manifest_path)
        if not mpath.exists():
            raise FileNotFoundError(f"Manifest not found: {mpath}")

        # parquet/csv loader (install pyarrow or fastparquet for parquet)
        if mpath.suffix.lower() == ".parquet":
            df = pd.read_parquet(mpath)
        else:
            df = pd.read_csv(mpath)

        # split
        if split is not None and "split" in df.columns:
            df = df[df["split"].astype(str).str.lower() == str(split).lower()].copy()

        # pandas query
        if filter_query:
            df = df.query(filter_query, engine="python")

        # regex include/exclude on root_dir
        if include_regex:
            rx_inc = re.compile(include_regex)
            df = df[df["root_dir"].astype(str).apply(lambda p: bool(rx_inc.search(p)))]
        if exclude_regex:
            rx_exc = re.compile(exclude_regex)
            df = df[~df["root_dir"].astype(str).apply(lambda p: bool(rx_exc.search(p)))]

        # numeric gates
        if min_slices is not None:
            df = df[df["n_slices"] >= int(min_slices)]
        if max_slices is not None:
            df = df[df["n_slices"] <= int(max_slices)]
        if max_size_8bit_GB is not None and "size_8bit_GB" in df.columns:
            df = df[df["size_8bit_GB"] <= float(max_size_8bit_GB)]

        # XML-driven gates (if present)
        if xml_kV_range is not None and "xml_kV" in df.columns:
            lo, hi = xml_kV_range
            df = df[df["xml_kV"].between(lo, hi, inclusive="both")]
        if vox_xy_range_mm is not None and {"xml_voxX","xml_voxY"}.issubset(df.columns):
            lo, hi = vox_xy_range_mm
            mxy = df[["xml_voxX","xml_voxY"]].astype(float).mean(axis=1)
            df = df[mxy.between(lo, hi, inclusive="both")]

        # per-group head-K
        if head_k_per_group is not None and group_col in df.columns:
            k = int(head_k_per_group)
            # sort by group, then a stable key (root_dir) to make selection deterministic
            df = (
                df.sort_values([group_col, "root_dir"])
                  .groupby(group_col, group_keys=False)
                  .apply(lambda g: g.head(k))
                  .reset_index(drop=True)
            )

        # random subset
        if random_subset is not None:
            n = len(df)
            if isinstance(random_subset, float):
                k = max(1, int(round(n * random_subset)))
            else:
                k = min(int(random_subset), n)
            df = df.sample(n=k, random_state=seed).reset_index(drop=True)

        if df.empty:
            raise RuntimeError("Manifest has no rows after filtering.")
        df = df.drop_duplicates(subset=["root_dir"]).reset_index(drop=True)
        self.df = df

        # materialize per-volume info (skip missing on disk)
        self.vols: List[VolumeInfo] = []
        for _, r in self.df.iterrows():
            root = Path(r["root_dir"])
            if not root.exists():
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
            raise RuntimeError("No valid volumes found on disk after filtering.")

        # sampling weights
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
            return self._rng.randrange(len(self.vols))
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
        return stack.contiguous().clone()

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

        # fetch slice list and dims
        slices = self._get_slice_paths(vidx)
        H0, W0 = v.H, v.W

        # choose spatial crop (reflect-pad to ensure valid region)
        def _rand_start(full, size):
            return 0 if full <= size else self._rng.randint(0, full - size)
        z0 = _rand_start(len(slices), D)
        y0 = _rand_start(H0, H)
        x0 = _rand_start(W0, W)

        # read and crop
        vol = self._read_stack(slices, z0=z0, D=D)  # (D,H0,W0)
        if H0 < H or W0 < W:
            pad_h = max(0, H - H0)
            pad_w = max(0, W - W0)
            vol = F.pad(
                vol.unsqueeze(0).unsqueeze(0),
                (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2, 0, 0),
                mode="reflect"
            ).squeeze(0).squeeze(0)
            H0, W0 = vol.shape[1], vol.shape[2]
            y0 = _rand_start(H0, H)
            x0 = _rand_start(W0, W)

        patch = vol[:, y0:y0+H, x0:x0+W].contiguous()  # (D,H,W)

        # normalize per manifest lo/hi
        sharp = self._norm01(patch, v.lo, v.hi)

        # blurred variant
        blurred = sharp.clone() if self.blur_transform is None else self.blur_transform(sharp)

        # ensure PyTorch-owned, resizable storages (not NumPy-backed)
        sharp   = sharp.contiguous().clone()
        blurred = blurred.contiguous().clone()
        return sharp, blurred

    # ---- convenience: quick summary in notebooks ----
    def summary(self, top=12):
        df = self.df
        cols = [c for c in ["root_dir","n_slices","H","W","size_8bit_GB","xml_kV","xml_voxX","xml_voxY"]
                if c in df.columns]
        out = df[cols].head(top).copy()
        total_gb = df["size_8bit_GB"].sum() if "size_8bit_GB" in df.columns else np.nan
        if isinstance(total_gb, float) and not np.isnan(total_gb):
            print(f"Scans kept: {len(df)}; 8-bit equiv total: {total_gb:.2f} GB")
        else:
            print(f"Scans kept: {len(df)}")
        return out
