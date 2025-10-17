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
    raise RuntimeError("tifffile is required. pip install tifffile") from e

__all__ = ["MultiPageTiffDataset"]


# -------------------------- helpers --------------------------
def _natural_key(name: str):
    num_re = re.compile(r"(\d+)")
    parts = num_re.split(name)
    parts[1::2] = [int(p) for p in parts[1::2]]
    return parts


def _probe_multipage_tif(path: Path) -> Tuple[int, int, int, str]:
    """
    Return (D, H, W, dtype_str) for a multi-page TIFF, by reading only the first page
    and counting pages (no full load).
    """
    with tiff.TiffFile(str(path)) as tf:
        D = len(tf.pages)
        first = tf.pages[0].asarray()
        if first.ndim != 2:
            raise ValueError(f"Expected 2D pages in multi-page TIFF: {path} got {first.shape}")
        H, W = first.shape
        dtype_str = str(first.dtype)
    return D, H, W, dtype_str


def _read_pages(path: Path, idxs: List[int]) -> torch.Tensor:
    """
    Read specific z-indices (pages) from a multi-page TIFF as float32 tensor (D, H, W) in [0..255].
    Only the requested pages are read.
    """
    imgs: List[torch.Tensor] = []
    with tiff.TiffFile(str(path)) as tf:
        n_pages = len(tf.pages)
        for i in idxs:
            if not (0 <= i < n_pages):
                raise IndexError(f"Page index {i} out of range 0..{n_pages-1} for {path}")
            arr = tf.pages[i].asarray()
            if arr.ndim != 2:
                raise ValueError(f"Non-2D page at z={i} in {path} shape={arr.shape}")
            imgs.append(torch.from_numpy(arr.astype(np.float32, copy=False)))
    return torch.stack(imgs, dim=0)


# -------------------------- dataclasses --------------------------
@dataclass
class VolumeInfo:
    tif_path: Path
    n_slices: int
    H: int
    W: int
    dtype_str: str  # typically 'uint8'
    spacing_y: Optional[float] = None
    spacing_x: Optional[float] = None


# -------------------------- dataset --------------------------
class MultiPageTiffDataset(Dataset):
    """
    Stream random 3D patches from *single multi-page TIFF* volumes listed in a manifest/index.

    Expected manifest columns (from your converter):
      - output_tif (path to .tif)
      - out_D, out_H, out_W  (optional; speeds up probing)
      - used_lo, used_hi     (ignored here; frames are already u8 scaled)
      - spacing_y, spacing_x (optional)

    If these columns are absent, the dataset will probe each TIFF once (cheap).

    Returns (sharp, blurred) as torch.float32 tensors in [0,1] shaped (D,H,W).

    Filters/subsetting (all optional):
      - filter_query: pandas query (e.g., "out_D>=64")
      - include_regex / exclude_regex: regex on output_tif
      - min_slices, max_slices
      - random_subset: int count or float fraction
      - samples_per_epoch: cap __len__ for faster epochs
      - balance: 'volume' or 'slice_count' (weights by out_D)
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: Optional[str] = None,              # kept for symmetry; ignored unless 'split' exists
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
        random_subset: Optional[int | float] = None,
        seed: int = 0,
        # deprecated passthrough for backwards compatibility with old notebooks:
        **kwargs
    ):
        super().__init__()
        for k in ("head_k_per_group", "group_col", "max_size_8bit_GB", "xml_kV_range", "vox_xy_range_mm"):
            if k in kwargs and kwargs[k] is not None:
                print(f"[MultiPageTiffDataset] Warning: '{k}' is deprecated/ignored for multi-page TIFFs.")

        self.patch_size = tuple(patch_size)
        self.blur_transform = blur_transform
        self.balance = balance
        self.samples_per_epoch = samples_per_epoch
        self._rng = random.Random(seed)

        mpath = Path(manifest_path)
        if not mpath.exists():
            raise FileNotFoundError(f"Manifest not found: {mpath}")

        # parquet/csv/xlsx support
        if mpath.suffix.lower() == ".parquet":
            df = pd.read_parquet(mpath)
        elif mpath.suffix.lower() in (".csv",):
            df = pd.read_csv(mpath)
        elif mpath.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(mpath)
        else:
            raise ValueError(f"Unsupported manifest format: {mpath.suffix}")

        # split
        if split is not None and "split" in df.columns:
            df = df[df["split"].astype(str).str.lower() == str(split).lower()].copy()

        # must have output_tif
        if "output_tif" not in df.columns:
            raise ValueError("Manifest must include 'output_tif' (path to the single multi-page TIFF).")

        # pandas query
        if filter_query:
            df = df.query(filter_query, engine="python")

        # regex include/exclude on output_tif
        if include_regex:
            rx_inc = re.compile(include_regex)
            df = df[df["output_tif"].astype(str).apply(lambda p: bool(rx_inc.search(p)))]
        if exclude_regex:
            rx_exc = re.compile(exclude_regex)
            df = df[~df["output_tif"].astype(str).apply(lambda p: bool(rx_exc.search(p)))]

        # drop non-existing tiffs
        df = df.copy()
        df["output_tif"] = df["output_tif"].astype(str)
        df["__exists__"] = df["output_tif"].apply(lambda p: Path(p).exists())
        df = df[df["__exists__"]].drop(columns=["__exists__"]).reset_index(drop=True)

        if df.empty:
            raise RuntimeError("No valid multi-page TIFFs found after filtering.")

        # random subset
        if random_subset is not None and len(df) > 0:
            n = len(df)
            if isinstance(random_subset, float):
                k = max(1, int(round(n * random_subset)))
            else:
                k = min(int(random_subset), n)
            df = df.sample(n=k, random_state=seed).reset_index(drop=True)

        if df.empty:
            raise RuntimeError("Empty dataset after subsetting.")

        # materialize per-volume info (prefer manifest dims if present)
        self.vols: List[VolumeInfo] = []
        for _, r in df.iterrows():
            tif_path = Path(r["output_tif"])
            if all(c in df.columns for c in ["out_D", "out_H", "out_W"]) and \
               pd.notna(r["out_D"]) and pd.notna(r["out_H"]) and pd.notna(r["out_W"]):
                D, H, W = int(r["out_D"]), int(r["out_H"]), int(r["out_W"])
                dtype_str = "uint8"  # your converter writes u8
            else:
                D, H, W, dtype_str = _probe_multipage_tif(tif_path)

            # numeric gating by slice count
            if min_slices is not None and D < int(min_slices):
                continue
            if max_slices is not None and D > int(max_slices):
                continue

            self.vols.append(VolumeInfo(
                tif_path=tif_path,
                n_slices=D,
                H=H, W=W,
                dtype_str=dtype_str,
                spacing_y=None if "spacing_y" not in r or pd.isna(r["spacing_y"]) else float(r["spacing_y"]),
                spacing_x=None if "spacing_x" not in r or pd.isna(r["spacing_x"]) else float(r["spacing_x"]),
            ))

        if not self.vols:
            raise RuntimeError("No volumes after applying slice-count filters.")

        # sampling weights
        if self.balance == "slice_count":
            arr = np.array([max(1, v.n_slices) for v in self.vols], dtype=np.float64)
            self.weights = (arr / arr.sum()).tolist()
        else:
            self.weights = None

    # ---------------- Dataset API ----------------
    def __len__(self) -> int:
        return self.samples_per_epoch if self.samples_per_epoch is not None else len(self.vols)

    def _pick_volume_index(self) -> int:
        if self.weights is None:
            return self._rng.randrange(len(self.vols))
        return int(np.random.choice(len(self.vols), p=self.weights))

    @torch.no_grad()
    def __getitem__(self, idx: int):
        # choose a volume (possibly weighted by D)
        vidx = self._pick_volume_index()
        v = self.vols[vidx]
        Dp, Hp, Wp = self.patch_size

        # robust Z indices (exactly Dp)
        n = v.n_slices
        if n >= Dp:
            z0 = self._rng.randint(0, n - Dp)
            z_idxs = list(range(z0, z0 + Dp))
        else:
            if n == 1:
                cycle = [0]
            else:
                cycle = list(range(n)) + list(range(n - 2, 0, -1))  # mirror cycle
            start = self._rng.randint(0, len(cycle) - 1)
            z_idxs = [cycle[(start + k) % len(cycle)] for k in range(Dp)]

        # spatial crop coord
        def _rand_start(full, size):  # inclusive
            return 0 if full <= size else self._rng.randint(0, full - size)

        y0 = _rand_start(v.H, Hp)
        x0 = _rand_start(v.W, Wp)
        pad_h = max(0, Hp - v.H)
        pad_w = max(0, Wp - v.W)

        # read only the selected pages, then crop/pad spatially
        vol = _read_pages(v.tif_path, z_idxs)  # (D, H, W), dtype float32 in [0..255]
        # crop
        vol = vol[:, y0:y0 + min(Hp, vol.shape[1]), x0:x0 + min(Wp, vol.shape[2])]
        # reflect pad to (Dp, Hp, Wp)
        if pad_h or pad_w:
            vol = F.pad(vol.unsqueeze(0).unsqueeze(0),
                        (pad_w // 2, pad_w - pad_w // 2,
                         pad_h // 2, pad_h - pad_h // 2,
                         0, 0), mode="reflect").squeeze(0).squeeze(0)
        vol = vol[:, :Hp, :Wp].contiguous()

        # normalize to [0,1] (already u8-scaled by your converter)
        sharp = (vol / 255.0).clamp_(0.0, 1.0)
        blurred = sharp.clone() if self.blur_transform is None else self.blur_transform(sharp)

        return sharp.contiguous().clone(), blurred.contiguous().clone()

    # ---------------- Convenience ----------------
    def summary(self, top=12):
        rows = []
        for v in self.vols[:top]:
            rows.append({
                "output_tif": str(v.tif_path),
                "out_D": v.n_slices, "out_H": v.H, "out_W": v.W,
                "dtype": v.dtype_str,
                "spacing_y": v.spacing_y, "spacing_x": v.spacing_x,
            })
        df = pd.DataFrame(rows)
        print(f"Scans kept: {len(self.vols)}")
        return df
