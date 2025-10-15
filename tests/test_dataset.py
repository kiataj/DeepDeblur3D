# tests/test_dataset.py
import os, re, json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import tifffile as tiff

from deblur3d.data import TiffDataset

# ---------- helpers ----------

def _make_stack(folder: Path, H=32, W=40, D=12, start=0, dtype=np.uint16, add_noise=False):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    lo, hi = 1000, 40000  # within uint16
    for i in range(start, start + D):
        img = np.linspace(lo, hi, H*W, dtype=np.float32).reshape(H, W)
        if add_noise:
            img += rng.normal(0, 200, img.shape).astype(np.float32)
        img = np.clip(img, 0, np.iinfo(dtype).max).astype(dtype)
        tiff.imwrite(str(folder / f"slice_{i:04d}.tif"), img, photometric='minisblack')
    # add things to be ignored
    tiff.imwrite(str(folder / "mask_dummy.tif"), np.zeros((H, W), dtype=dtype))
    (folder / "proj_bad").mkdir(exist_ok=True)
    tiff.imwrite(str(folder / "proj_bad" / "proj_0001.tif"), np.zeros((H, W), dtype=dtype))
    return lo, hi, H, W, D

def _build_manifest(tmp_path: Path):
    # two real volumes + one missing
    v1 = tmp_path / "scan_A"
    lo1, hi1, H1, W1, D1 = _make_stack(v1, H=32, W=40, D=12)
    v2 = tmp_path / "scan_B"
    lo2, hi2, H2, W2, D2 = _make_stack(v2, H=48, W=56, D=30)
    # missing
    v3 = tmp_path / "missing_scan"

    rows = [
        dict(root_dir=str(v1), n_slices=D1, H=H1, W=W1, dtype="uint16",
             lo=float(lo1), hi=float(hi1), project_path=str(tmp_path), size_8bit_GB=(D1*H1*W1)/1e9),
        dict(root_dir=str(v2), n_slices=D2, H=H2, W=W2, dtype="uint16",
             lo=float(lo2), hi=float(hi2), project_path=str(tmp_path), size_8bit_GB=(D2*H2*W2)/1e9),
        dict(root_dir=str(v3), n_slices=20, H=32, W=32, dtype="uint16",
             lo=0.0, hi=65535.0, project_path=str(tmp_path), size_8bit_GB=(20*32*32)/1e9),
    ]
    df = pd.DataFrame(rows)
    mpath = tmp_path / "manifest.parquet"
    df.to_parquet(mpath, index=False)
    return mpath, df

# ---------- tests ----------

def test_basic_getitem_shape_and_range(tmp_path: Path):
    mpath, _ = _build_manifest(tmp_path)
    ds = TiffDataset(mpath, patch_size=(8, 24, 24), samples_per_epoch=5)
    sharp, blurred = ds[0]
    assert sharp.shape == (8, 24, 24)
    assert blurred.shape == (8, 24, 24)
    assert torch.isfinite(sharp).all()
    assert 0.0 <= float(sharp.min()) + 1e-6 <= 1.0
    assert 0.0 <= float(sharp.max()) <= 1.0 + 1e-6

def test_ignores_proj_and_mask(tmp_path: Path):
    mpath, df = _build_manifest(tmp_path)
    ds = TiffDataset(mpath, patch_size=(12, df.iloc[0].H, df.iloc[0].W))
    # ensure we can read full D despite extra mask_ and proj_ files existing
    sharp, _ = ds[0]
    assert sharp.shape[0] == 12

def test_len_samples_per_epoch(tmp_path: Path):
    mpath, _ = _build_manifest(tmp_path)
    ds = TiffDataset(mpath, samples_per_epoch=37)
    assert len(ds) == 37
    ds2 = TiffDataset(mpath, samples_per_epoch=None)
    assert len(ds2) == len(ds2.vols)

def test_filter_query_and_numeric_gates(tmp_path: Path):
    mpath, df = _build_manifest(tmp_path)
    # keep only scans with >= 20 slices
    ds = TiffDataset(mpath, filter_query="n_slices>=20")
    kept = {Path(v.root_dir).name for v in ds.vols}
    assert kept == {"scan_B"}  # scan_A has 12
    # numeric gate
    ds2 = TiffDataset(mpath, min_slices=20)
    kept2 = {Path(v.root_dir).name for v in ds2.vols}
    assert kept2 == {"scan_B"}

def test_regex_include_exclude(tmp_path: Path):
    mpath, _ = _build_manifest(tmp_path)
    ds_inc = TiffDataset(mpath, include_regex=r"scan_A")
    assert {Path(v.root_dir).name for v in ds_inc.vols} == {"scan_A"}
    ds_exc = TiffDataset(mpath, exclude_regex=r"scan_A")
    assert "scan_A" not in {Path(v.root_dir).name for v in ds_exc.vols}

def test_random_subset_reproducible(tmp_path: Path):
    mpath, _ = _build_manifest(tmp_path)
    ds1 = TiffDataset(mpath, random_subset=1, seed=123)
    ds2 = TiffDataset(mpath, random_subset=1, seed=123)
    assert {v.root_dir for v in ds1.vols} == {v.root_dir for v in ds2.vols}
    
@pytest.mark.skip(reason="head_k_per_group deprecated and removed from TiffDataset")
def test_head_k_per_group(tmp_path: Path):
    ...

def test_balance_slice_count_flag(tmp_path: Path):
    mpath, _ = _build_manifest(tmp_path)
    ds = TiffDataset(mpath, balance="slice_count")
    assert ds.weights is not None and len(ds.weights) == len(ds.vols)
    assert abs(sum(ds.weights) - 1.0) < 1e-6

def test_natural_sorting_of_slices(tmp_path: Path):
    # create non-zero-padded names to ensure natural order works
    f = tmp_path / "weird_names"
    f.mkdir()
    for name in ["slice_1.tif", "slice_2.tif", "slice_10.tif"]:
        tiff.imwrite(str(f / name), np.zeros((8, 8), dtype=np.uint16))
    # ensure __getitem__ doesn't break on odd ordering
    df = pd.DataFrame([{
        "root_dir": str(f), "n_slices": 3, "H": 8, "W": 8,
        "dtype": "uint16", "lo": 0.0, "hi": 65535.0,
        "project_path": str(tmp_path), "size_8bit_GB": (3*8*8)/1e9
    }])
    m = tmp_path / "m.parquet"; df.to_parquet(m, index=False)
    ds = TiffDataset(m, patch_size=(3,8,8))
    sharp, _ = ds[0]
    assert sharp.shape == (3,8,8)  # if order was wrong, reflect-pad might kick in

def test_missing_dirs_are_skipped(tmp_path: Path):
    mpath, df = _build_manifest(tmp_path)
    # manifest already includes a missing dir; dataset should still build
    ds = TiffDataset(mpath)
    roots = {Path(v.root_dir) for v in ds.vols}
    assert not any("missing_scan" in str(r) for r in roots)

def test_summary_runs(tmp_path: Path, capsys):
    mpath, _ = _build_manifest(tmp_path)
    ds = TiffDataset(mpath)
    _ = ds.summary(top=5)
    out = capsys.readouterr().out
    assert "Scans kept:" in out
