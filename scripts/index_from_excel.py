#!/usr/bin/env python
# scripts/index_from_excel.py
import argparse, os, sys, re, json, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import tifffile as tiff
except Exception as e:
    print("ERROR: tifffile is required. pip install tifffile", file=sys.stderr)
    raise

NUM_RE = re.compile(r'(\d+)')

import xml.etree.ElementTree as ET

# dtype → bytes-per-pixel
_DTYPE_BPP = {
    "uint8":1, "int8":1,
    "uint16":2, "int16":2,
    "uint32":4, "int32":4,
    "float32":4, "single":4,
    "float64":8, "double":8,
}
def bpp_from_dtype_str(dtype_str: str) -> int:
    s = str(dtype_str).lower()
    for k, v in _DTYPE_BPP.items():
        if k in s: return v
    return 2  # sensible fallback

def _get_attr(elem, key, cast=str):
    if elem is None: return None
    v = elem.attrib.get(key)
    if v is None: return None
    try:
        return cast(v)
    except Exception:
        try:
            # some strings like "16 bit Tiff" -> grab leading number if cast=int
            if cast is int:
                import re
                m = re.search(r'(\d+)', v)
                return int(m.group(1)) if m else None
            if cast is float:
                return float(str(v).replace(',', '.'))
        except Exception:
            pass
    return v  # fallback raw

def parse_unireconstruction(xml_path: Path) -> dict:
    """Parse unireconstruction.xml into a flat dict (xml_* keys)."""
    out = {}
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        cb = root.find("conebeam")
        if cb is None:
            return out

        # reconstruct
        rec = cb.find("reconstruct")
        out["xml_method"]      = _get_attr(rec, "methode", str)
        out["xml_filter"]      = _get_attr(rec, "filter", str)
        out["xml_filterValue"] = _get_attr(rec, "filterValue", float)
        out["xml_appodizationFilter"] = _get_attr(rec, "appodizationFilter", str)
        out["xml_appodization"] = _get_attr(rec, "appodization", float)
        out["xml_format"]      = _get_attr(rec, "format", str)
        out["xml_zfileformat"] = _get_attr(rec, "zfileformat", str)
        out["xml_iterations"]  = _get_attr(rec, "iterations", int)

        # volume
        vol = cb.find("volume")
        vvox = vol.find("voxelSize") if vol is not None else None
        vsize= vol.find("size")      if vol is not None else None
        voff = vol.find("offset")    if vol is not None else None
        vrot = vol.find("rotate")    if vol is not None else None

        out["xml_voxX"] = _get_attr(vvox, "X", float)
        out["xml_voxY"] = _get_attr(vvox, "Y", float)
        out["xml_voxZ"] = _get_attr(vvox, "Z", float)

        out["xml_sizeX"] = _get_attr(vsize, "X", int)
        out["xml_sizeY"] = _get_attr(vsize, "Y", int)
        out["xml_sizeZ"] = _get_attr(vsize, "Z", int)

        out["xml_offX"]  = _get_attr(voff, "X", float)
        out["xml_offY"]  = _get_attr(voff, "Y", float)
        out["xml_offZ"]  = _get_attr(voff, "Z", float)

        out["xml_rotX"]  = _get_attr(vrot, "X", float)
        out["xml_rotY"]  = _get_attr(vrot, "Y", float)
        out["xml_rotZ"]  = _get_attr(vrot, "Z", float)

        # acquisition/profile
        prof = cb.find("profile")
        out["xml_images"] = _get_attr(prof, "images", int)
        xray = cb.find("acquisitioninfo/xray")
        out["xml_kV"]     = _get_attr(xray, "voltage", float)
        out["xml_uA"]     = _get_attr(xray, "current", float)

        geom = cb.find("acquisitioninfo/geometry")
        out["xml_SOD"]    = _get_attr(geom, "sod", float)
        out["xml_SDD"]    = _get_attr(geom, "sdd", float)

        # optional toggles
        ring = cb.find("ring_filter")
        out["xml_ring_enabled"] = _get_attr(ring, "ringEnabled", int)

        phase = cb.find("phase_filter")
        out["xml_phase_enabled"] = _get_attr(phase, "enabled", int)
        out["xml_phase_type"]    = _get_attr(phase, "type", str)
        out["xml_phase_param"]   = _get_attr(phase, "manu_para", float)

        # derived
        try:
            vx = float(out["xml_voxX"]) if out["xml_voxX"] is not None else None
            vy = float(out["xml_voxY"]) if out["xml_voxY"] is not None else None
            vz = float(out["xml_voxZ"]) if out["xml_voxZ"] is not None else None
            sx = int(out["xml_sizeX"])  if out["xml_sizeX"] is not None else None
            sy = int(out["xml_sizeY"])  if out["xml_sizeY"] is not None else None
            sz = int(out["xml_sizeZ"])  if out["xml_sizeZ"] is not None else None
            if all(v is not None for v in (vx,vy,vz,sx,sy,sz)):
                out["xml_physX"] = vx * sx
                out["xml_physY"] = vy * sy
                out["xml_physZ"] = vz * sz
                out["xml_anisotropy"] = max(vx,vy,vz) / max(1e-12, min(vx,vy,vz))
                out["xml_bytes_8bit_from_xml"] = int(sx) * int(sy) * int(sz)  # 1 byte/voxel
        except Exception:
            pass

    except Exception:
        # keep silent at row level; caller can log
        return out

    return out


def natural_key(s: str):
    parts = NUM_RE.split(s)
    parts[1::2] = [int(p) for p in parts[1::2]]
    return parts

def list_tifs(folder: Path) -> List[Path]:
    # case-insensitive *.tif / *.tiff
    exts = (".tif", ".tiff")
    files = [p for p in folder.iterdir()
             if p.is_file()
             and p.suffix.lower() in exts
             and not p.name.lower().startswith("proj_")
             and not p.name.lower().startswith("mask_")]
    files.sort(key=lambda p: natural_key(p.name))
    return files

def peek_slices_and_stats(files: List[Path], sample_n: int = 16) -> Tuple[int, int, str, float, float]:
    """Return (H, W, dtype_str, lo, hi) using up to 'sample_n' evenly-spaced slices."""
    n = len(files)
    idxs = np.linspace(0, n - 1, num=min(sample_n, n), dtype=int)
    lo_vals, hi_vals = [], []
    H = W = None
    dtype = None
    for i in idxs:
        arr = tiff.imread(str(files[i]))
        if arr.ndim != 2:
            raise ValueError(f"Slice {files[i]} not 2D (got {arr.shape})")
        if H is None:
            H, W = arr.shape
            dtype = str(arr.dtype)
        else:
            if arr.shape != (H, W):
                raise ValueError(f"Inconsistent shapes in sequence: {arr.shape} vs {(H,W)}")
        a = arr.astype(np.float32)
        lo_vals.append(np.percentile(a, 1.0))
        hi_vals.append(np.percentile(a, 99.5))
    lo = float(np.median(lo_vals)) if lo_vals else 0.0
    hi = float(np.median(hi_vals)) if hi_vals else 1.0
    if hi <= lo:
        hi = lo + 1e-3
    return H, W, dtype, lo, hi

def try_read_xy_resolution(path: Path) -> Tuple[Optional[float], Optional[float]]:
    """Best-effort TIFF XY pixel size (in inverse of resolution unit). Returns (spacing_y, spacing_x) or (None,None)."""
    try:
        with tiff.TiffFile(str(path)) as tf:
            page = tf.pages[0]
            tags = page.tags
            xres = tags.get("XResolution")
            yres = tags.get("YResolution")
            runit = tags.get("ResolutionUnit")
            if xres and yres:
                # XResolution/YResolution often stored as rationals (num, den)
                def _as_float(v):
                    try:
                        return float(v.value[0] / v.value[1]) if hasattr(v.value, "__len__") else float(v.value)
                    except Exception:
                        try:
                            num, den = v.value
                            return float(num) / float(den)
                        except Exception:
                            return None
                xr = _as_float(xres)
                yr = _as_float(yres)
                if xr and yr and xr > 0 and yr > 0:
                    # resolution is typically in pixels per unit (e.g., per inch). We invert to get size per pixel (unit/pixel).
                    spx = 1.0 / xr
                    spy = 1.0 / yr
                    # We do not convert physical units (inch/mm); keep as given to be consistent across the manifest.
                    return float(spy), float(spx)
    except Exception:
        pass
    return None, None

def scan_project(project_path: Path, min_slices: int, sample_slices: int) -> Tuple[List[Dict], List[Dict]]:
    """Return (rows, logs) for this project."""
    rows, logs = [], []
    if not project_path.exists():
        logs.append({"level": "error", "project": str(project_path), "msg": "path does not exist"})
        return rows, logs

    for dirpath, dirnames, filenames in os.walk(project_path):
        base = os.path.basename(dirpath)
        if base.lower().startswith("proj_"):
            # skip projections folders entirely
            dirnames[:] = []  # do not descend into this tree
            continue

        folder = Path(dirpath)
        tifs = list_tifs(folder)
        if len(tifs) < min_slices:
            continue

        try:
            # --- basic slice stats ---
            H, W, dtype, lo, hi = peek_slices_and_stats(tifs, sample_n=sample_slices)
            # try XY spacing from the first slice
            sy, sx = try_read_xy_resolution(tifs[0])

            # --- sizes ---
            # 8-bit equivalent: 1 byte/voxel
            bytes_8bit = int(len(tifs)) * int(H) * int(W)

            # Uncompressed estimate from dtype (bytes per pixel)
            bpp = bpp_from_dtype_str(dtype)  # helper: maps dtype string to 1/2/4/8
            bytes_est = int(len(tifs)) * int(H) * int(W) * int(bpp)

            # Exact on-disk size (may be slower but reliable with TIFF compression)
            try:
                bytes_exact = sum(f.stat().st_size for f in tifs)
            except Exception as e:
                logs.append({"level": "warn", "scan_dir": str(folder), "msg": f"exact size failed: {e}"})
                bytes_exact = None

            # --- XML metadata from parent folder ---
            xml_meta = {}
            try:
                xml_path = folder.parent / "unireconstruction.xml"
                if not xml_path.exists():
                    # try case-insensitive fallback
                    cand = next((p for p in folder.parent.glob("*.xml")
                                 if p.name.lower() == "unireconstruction.xml"), None)
                    xml_path = cand if cand else xml_path
                if xml_path and xml_path.exists():
                    xml_meta = parse_unireconstruction(xml_path)  # returns dict with xml_* keys
            except Exception as e:
                logs.append({"level": "warn", "scan_dir": str(folder), "msg": f"xml parse failed: {e}"})

            # --- row ---
            row = {
                "volume_id": f"{folder.as_posix()}",
                "project_path": str(project_path),
                "root_dir": str(folder),
                "n_slices": int(len(tifs)),
                "H": int(H), "W": int(W), "dtype": dtype,
                "lo": lo, "hi": hi,
                "spacing_y": sy, "spacing_x": sx,

                # sizes: 8-bit equivalent (always meaningful)
                "bytes_8bit": int(bytes_8bit),
                "size_8bit_GB": float(bytes_8bit / 1_000_000_000),
                "size_8bit_TB": float(bytes_8bit / 1_000_000_000_000),
                "size_8bit_TiB": float(bytes_8bit / (1024**4)),

                # sizes: dtype-based uncompressed estimate
                "bytes_est": int(bytes_est),
                "size_est_GB": float(bytes_est / 1_000_000_000),

                # sizes: exact on-disk (may be None if stat failed)
                "bytes_exact": None if bytes_exact is None else int(bytes_exact),
                "size_exact_GB": None if bytes_exact is None else float(bytes_exact / 1_000_000_000),
            }
            # merge XML fields (prefixed with xml_)
            if xml_meta:
                row.update(xml_meta)

            rows.append(row)

        except Exception as e:
            logs.append({"level": "warn", "scan_dir": str(folder), "msg": str(e)})
            continue

    if not rows:
        logs.append({"level": "info", "project": str(project_path), "msg": "no scan folders found"})
    return rows, logs


def pick_path_column(df: pd.DataFrame, explicit: Optional[str]) -> str:
    if explicit and explicit in df.columns:
        return explicit
    # Heuristic: first column that looks like a path and has existing dirs.
    for col in df.columns:
        if df[col].dtype == object:
            vals = df[col].dropna().astype(str)
            if len(vals) == 0:
                continue
            # Check a small sample
            ok = 0
            for v in vals.sample(min(10, len(vals)), random_state=0):
                p = Path(v)
                if p.exists() and p.is_dir():
                    ok += 1
            if ok >= max(3, min(3, len(vals))):  # at least 3 existing dirs
                return col
    raise ValueError("Could not infer project path column. Use --col to specify it explicitly.")

def main():
    ap = argparse.ArgumentParser(description="Index CT scan folders (TIFF sequences) from an Excel of project paths.")
    ap.add_argument("--excel", required=True, help="Path to Excel file listing project folders.")
    ap.add_argument("--col", default=None, help="Column name containing project folder paths (if omitted, auto-detect).")
    ap.add_argument("--out", default="manifest.parquet", help="Output manifest path (parquet preferred; csv fallback).")
    ap.add_argument("--min-slices", type=int, default=16, help="Minimum TIFF slices to accept a folder as a scan.")
    ap.add_argument("--sample-slices", type=int, default=16, help="Slices to sample for stats/validation per scan.")
    ap.add_argument("--workers", type=int, default=8, help="Parallel workers for indexing.")
    args = ap.parse_args()

    df = pd.read_excel(args.excel)
    col = pick_path_column(df, args.col)
    projects = [Path(p) for p in df[col].dropna().astype(str).tolist()]

    all_rows, all_logs = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(scan_project, p, args.min_slices, args.sample_slices): p for p in projects}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Indexing projects"):
            rows, logs = fut.result()
            all_rows.extend(rows)
            all_logs.extend(logs)

    if not all_rows:
        print("No volumes found. Exiting.", file=sys.stderr)
        sys.exit(1)

    manifest = pd.DataFrame(all_rows)
    manifest = manifest.drop_duplicates(subset=["root_dir"]).reset_index(drop=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Save manifest: parquet if possible, else csv
    saved_as = None
    try:
        import pyarrow  # noqa: F401
        manifest.to_parquet(out, index=False)
        saved_as = out
    except Exception:
        csv_path = out.with_suffix(".csv")
        manifest.to_csv(csv_path, index=False)
        saved_as = csv_path
        print(f"WARNING: pyarrow not found; wrote CSV instead: {csv_path}", file=sys.stderr)

    # Save logs
    log_path = out.with_suffix(".index_log.jsonl")
    with open(log_path, "w", encoding="utf-8") as f:
        for rec in all_logs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Indexed {len(manifest)} volumes.")
    print(f"Manifest: {saved_as}")
    print(f"Log:      {log_path}")

if __name__ == "__main__":
    main()
