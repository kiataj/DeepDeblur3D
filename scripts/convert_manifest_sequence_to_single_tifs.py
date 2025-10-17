#!/usr/bin/env python
# convert_manifest_sequences_to_single_tifs.py
# Requirements: pip install tifffile pandas numpy tqdm XlsxWriter pyarrow

import os, math, warnings, argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import tifffile as tiff
except Exception as e:
    raise RuntimeError("This script requires 'tifffile'. Install: pip install tifffile") from e


# -------------------- helpers --------------------
def _list_slices(scan_dir: Path) -> List[Path]:
    """
    List 2D TIFF slices in natural order, ignoring files starting with 'proj_' or 'mask_' (case-insensitive).
    """
    import re
    def natural_key(name: str):
        parts = re.split(r'(\d+)', name)
        parts[1::2] = [int(p) for p in parts[1::2]]
        return parts

    exts = (".tif", ".tiff")
    files = [p for p in scan_dir.iterdir()
             if p.is_file()
             and p.suffix.lower() in exts
             and not p.name.lower().startswith(("proj_", "mask_"))]
    files.sort(key=lambda p: natural_key(p.name))
    return files


def _estimate_lo_hi_from_sample(paths: List[Path],
                                sample_slices: int = 64,
                                percentiles: Tuple[float, float] = (1.0, 99.9)) -> Tuple[float, float]:
    """
    Robust lo/hi from a subset of slices to avoid reading entire volume.
    """
    if not paths:
        return 0.0, 1.0
    idx = np.linspace(0, len(paths) - 1, num=min(len(paths), sample_slices), dtype=int)
    samp = []
    for i in idx:
        arr = tiff.imread(str(paths[i]))
        if arr.ndim != 2:
            raise ValueError(f"Slice {paths[i]} is not 2D, got shape {arr.shape}")
        samp.append(arr.astype(np.float32, copy=False))
    block = np.stack(samp, axis=0)
    lo, hi = np.percentile(block, percentiles).tolist()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(block.min()), float(block.max())
    return float(lo), float(hi)


def _scale_to_u8(arr_f32: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Scale float32 array to uint8 using lo/hi, clipping to [0, 255].
    """
    y = (arr_f32 - lo) / max(hi - lo, 1e-6)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8, copy=False)


def _pool2d_sum_u8(u8: np.ndarray, factor: int) -> np.ndarray:
    """
    Sum non-overlapping (factor x factor) blocks on a uint8 2D array → uint32.
    Used to accumulate spatial block sums before dividing to get mean.
    """
    if factor <= 1:
        return u8.astype(np.uint32, copy=False)
    H, W = u8.shape
    HH = (H // factor) * factor
    WW = (W // factor) * factor
    if HH == 0 or WW == 0:
        return u8.astype(np.uint32, copy=False)
    x = u8[:HH, :WW].reshape(HH // factor, factor, WW // factor, factor)
    return x.sum(axis=(1, 3), dtype=np.uint32)


def _downsample3d_streaming_write(
    slices: List[Path],
    tw,
    ds: int,
    scale_lo: float,
    scale_hi: float,
    write_compression: Optional[str],
) -> Tuple[int, int, int]:
    """
    Stream TRUE 3D mean pooling over (ds, ds, ds) and write to TiffWriter `tw`.
    Returns (D_out, H_out, W_out).
    - XY are cropped to multiples of ds for exact block reduce.
    - Z uses grouped averaging; the last group may be smaller than ds (handled correctly).
    """
    # Probe shape
    s0 = tiff.imread(str(slices[0]))
    if s0.ndim != 2:
        raise ValueError(f"Expected 2D slices; got shape={s0.shape}")
    H, W = s0.shape
    D = len(slices)

    if ds <= 1:
        # No downsampling; write scaled u8 directly
        for p in slices:
            arr = tiff.imread(str(p)).astype(np.float32, copy=False)
            u8 = _scale_to_u8(arr, scale_lo, scale_hi)
            tw.write(u8, photometric="minisblack", compression=write_compression, contiguous=False)
        return D, H, W

    # Crop XY to multiples of ds for exact block reduce
    HH = (H // ds) * ds
    WW = (W // ds) * ds
    if HH == 0 or WW == 0:
        # Too small → fallback to no downsample
        for p in slices:
            arr = tiff.imread(str(p)).astype(np.float32, copy=False)
            u8 = _scale_to_u8(arr, scale_lo, scale_hi)
            tw.write(u8, photometric="minisblack", compression=write_compression, contiguous=False)
        return D, H, W

    H_out, W_out = HH // ds, WW // ds
    acc = np.zeros((H_out, W_out), dtype=np.uint32)  # accumulates spatial block sums across Z
    zcount = 0
    D_out = 0

    for i, p in enumerate(slices, start=1):
        arr = tiff.imread(str(p)).astype(np.float32, copy=False)
        u8 = _scale_to_u8(arr, scale_lo, scale_hi)
        acc += _pool2d_sum_u8(u8, ds)
        zcount += 1

        # When we collected ds slices, or at the tail, emit one pooled slice
        if zcount == ds or i == len(slices):
            denom = ds * ds * zcount
            out = ((acc.astype(np.float32) / float(denom)) + 0.5).astype(np.uint8, copy=False)
            tw.write(out, photometric="minisblack", compression=write_compression, contiguous=False)
            acc.fill(0)
            zcount = 0
            D_out += 1

    return D_out, H_out, W_out


def _parse_tuple_float(s: Optional[str], default: Tuple[float, float]) -> Tuple[float, float]:
    if not s:
        return default
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Expected two comma-separated floats, e.g. '1.0,99.9'")
    return float(parts[0]), float(parts[1])


def _parse_tuple_int(s: Optional[str], default: Tuple[int, int]) -> Tuple[int, int]:
    if not s:
        return default
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Expected two comma-separated ints, e.g. '2,4'")
    return int(parts[0]), int(parts[1])


# -------------------- main converter --------------------
def convert_manifest_sequences_to_single_tifs(
    manifest_path: str | Path,
    out_dir: str | Path,
    *,
    filter_query: Optional[str] = None,        # e.g. "n_slices >= 128 and size_8bit_GB < 6"
    id_prefix: str = "S",
    start_id: int = 1,
    overwrite: bool = False,
    sample_slices_for_range: int = 64,
    range_percentiles: Tuple[float, float] = (1.0, 99.9),
    bigtiff_threshold_gb: float = 3.5,         # BigTIFF if projected size exceeds this
    write_compression: Optional[str] = None,   # e.g. "zlib" (smaller, slower). None = fastest
    excel_name: str = "single_tif_index.xlsx",

    # Size-based TRUE 3D downsampling policy
    ds_thresholds_gb: Tuple[float, float] = (0.4, 1.0),   # (>0.4→2×, >1.0→4×)
    ds_factors: Tuple[int, int] = (2, 4),                 # corresponding factors
) -> Path:
    """
    Convert per-scan TIFF slice folders (from a manifest) into single 8-bit multi-page TIFFs.
    Adds automatic TRUE 3D downsampling via mean pooling (ds×ds×ds) when large.

      if size_8bit_GB > ds_thresholds_gb[1] -> ds_factors[1] (default 4×)
      elif size_8bit_GB > ds_thresholds_gb[0] -> ds_factors[0] (default 2×)
      else -> 1×

    'size_8bit_GB' is taken from the manifest if present; otherwise estimated as D*H*W / 1e9 (u8).
    """
    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    if manifest_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(manifest_path)
    else:
        df = pd.read_csv(manifest_path)

    if filter_query:
        try:
            df = df.query(filter_query, engine="python")
        except Exception as e:
            raise ValueError(f"Invalid filter_query: {filter_query}\n{e}")

    if df.empty:
        raise RuntimeError("No rows after filtering—nothing to convert.")

    # Required columns
    required = {"root_dir"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    next_id = int(start_id)
    records: List[Dict] = []

    thr_lo, thr_hi = ds_thresholds_gb
    f2, f4 = ds_factors

    # Process scans
    for _, row in tqdm(df.reset_index(drop=True).iterrows(), total=len(df), desc="Converting scans"):
        scan_dir = Path(row["root_dir"])
        if not scan_dir.exists():
            warnings.warn(f"[skip] Missing scan folder: {scan_dir}")
            continue

        slices = _list_slices(scan_dir)
        if len(slices) == 0:
            warnings.warn(f"[skip] No tif slices in: {scan_dir}")
            continue

        # Shape probe
        s0 = tiff.imread(str(slices[0]))
        if s0.ndim != 2:
            warnings.warn(f"[skip] First slice not 2D in: {scan_dir} (shape={s0.shape})")
            continue
        H, W = s0.shape
        D = len(slices)

        # lo/hi (from manifest if present, else estimate)
        lo = float(row["lo"]) if "lo" in row and pd.notna(row["lo"]) else None
        hi = float(row["hi"]) if "hi" in row and pd.notna(row["hi"]) else None
        if lo is None or hi is None:
            lo, hi = _estimate_lo_hi_from_sample(
                slices, sample_slices=sample_slices_for_range, percentiles=range_percentiles
            )

        # Size and downsample factor
        if "size_8bit_GB" in row and pd.notna(row["size_8bit_GB"]):
            size_gb_8bit = float(row["size_8bit_GB"])
        else:
            size_gb_8bit = (D * H * W) / 1e9

        if size_gb_8bit > thr_hi:
            ds = int(f4)
        elif size_gb_8bit > thr_lo:
            ds = int(f2)
        else:
            ds = 1

        # Predict post-downsample dims/sizes (for BigTIFF decision)
        if ds <= 1:
            D_out_pred, H_out_pred, W_out_pred = D, H, W
        else:
            D_out_pred = math.ceil(D / ds)
            H_out_pred = H // ds
            W_out_pred = W // ds

        projected_gb_after = (D_out_pred * H_out_pred * W_out_pred) / 1e9
        bigtiff_flag = projected_gb_after > bigtiff_threshold_gb

        # Output path
        scan_id = f"{id_prefix}{next_id:04d}"
        out_path = out_dir / f"{scan_id}.tif"
        next_id += 1

        if out_path.exists() and not overwrite:
            warnings.warn(f"[skip] Exists: {out_path} (use overwrite=True to rewrite)")
            # record mapping even when skipped
            rec = dict(row)
            rec.update({
                "scan_id": scan_id,
                "output_tif": str(out_path),
                "out_H": int(H_out_pred), "out_W": int(W_out_pred), "out_D": int(D_out_pred),
                "used_lo": lo, "used_hi": hi,
                "ds_factor": ds,
                "projected_gb_before": size_gb_8bit,
                "projected_gb_after": projected_gb_after,
                "status": "skipped_exists",
            })
            records.append(rec)
            continue

        # Write with TRUE 3D downsampling (streamed)
        try:
            with tiff.TiffWriter(str(out_path), bigtiff=bigtiff_flag) as tw:
                D_written, H_written, W_written = _downsample3d_streaming_write(
                    slices=slices,
                    tw=tw,
                    ds=ds,
                    scale_lo=lo,
                    scale_hi=hi,
                    write_compression=write_compression,
                )
        except Exception as e:
            warnings.warn(f"[error] Failed writing {out_path}: {e}")
            continue

        rec = dict(row)
        rec.update({
            "scan_id": scan_id,
            "output_tif": str(out_path),
            "out_H": int(H_written), "out_W": int(W_written), "out_D": int(D_written),
            "used_lo": lo, "used_hi": hi,
            "ds_factor": ds,
            "projected_gb_before": size_gb_8bit,
            "projected_gb_after": projected_gb_after,
            "status": "ok",
        })
        records.append(rec)

    # Build Excel index
    if not records:
        raise RuntimeError("No scans converted—check your filter or manifest.")

    df_out = pd.DataFrame(records)
    excel_path = out_dir / excel_name
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as xw:
        df_out.to_excel(xw, index=False, sheet_name="index")

    print(f"\nDone. Wrote {len(df_out[df_out['status']=='ok'])} scans to: {out_dir}")
    print(f"Excel index: {excel_path}")
    return excel_path


# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Convert TIFF slice folders (from manifest) to single 8-bit multipage TIFFs with optional 3D downsampling.")
    ap.add_argument("--manifest", required=True, help="Path to manifest (.parquet or .csv)")
    ap.add_argument("--out-dir", required=True, help="Output directory for single TIFFs")
    ap.add_argument("--filter", default=None, help="Optional pandas query to subset rows")
    ap.add_argument("--id-prefix", default="S", help="Prefix for output scan IDs (default: S)")
    ap.add_argument("--start-id", type=int, default=1, help="Starting integer for scan IDs (default: 1)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    ap.add_argument("--sample-slices", type=int, default=64, help="Slices to sample for lo/hi estimate (default: 64)")
    ap.add_argument("--range-percentiles", default="1.0,99.9", help="Percentiles for lo,hi (e.g., '1.0,99.9')")
    ap.add_argument("--bigtiff-threshold-gb", type=float, default=3.5, help="Switch to BigTIFF if projected size exceeds this (GB)")
    ap.add_argument("--compression", default=None, help="TIFF compression: None, 'zlib', 'lzma', etc.")
    ap.add_argument("--ds-thresholds-gb", default="0.4,1.0", help="Size thresholds in GB (e.g., '0.4,1.0')")
    ap.add_argument("--ds-factors", default="2,4", help="Downsample factors (e.g., '2,4')")
    ap.add_argument("--excel-name", default="single_tif_index.xlsx", help="Output Excel index filename")
    args = ap.parse_args()

    range_percentiles = _parse_tuple_float(args.range_percentiles, (1.0, 99.9))
    ds_thresholds_gb = _parse_tuple_float(args.ds_thresholds_gb, (0.4, 1.0))
    ds_factors = _parse_tuple_int(args.ds_factors, (2, 4))

    convert_manifest_sequences_to_single_tifs(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        filter_query=args.filter,
        id_prefix=args.id_prefix,
        start_id=args.start_id,
        overwrite=args.overwrite,
        sample_slices_for_range=args.sample_slices,
        range_percentiles=range_percentiles,
        bigtiff_threshold_gb=args.bigtiff_threshold_gb,
        write_compression=(None if args.compression in (None, "", "None", "none") else args.compression),
        excel_name=args.excel_name,
        ds_thresholds_gb=ds_thresholds_gb,
        ds_factors=ds_factors,
    )


if __name__ == "__main__":
    main()
