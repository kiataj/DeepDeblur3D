#!/usr/bin/env python
# split_train_val_hash.py
# Deterministic, group-aware train/val split using a stable hash on group keys.
# Deps: pandas (and pyarrow for parquet; openpyxl/XlsxWriter for xlsx)

import argparse, hashlib
from pathlib import Path
import pandas as pd


def stable_hash(x: str) -> int:
    """MD5-based stable hash → int (deterministic across runs/platforms)."""
    return int(hashlib.md5(x.encode("utf-8")).hexdigest(), 16)


def _load_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    elif suf == ".csv":
        return pd.read_csv(path)
    elif suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {suf}")


def _save_table(df: pd.DataFrame, out_path: Path) -> None:
    suf = out_path.suffix.lower()
    if suf == ".parquet":
        df.to_parquet(out_path, index=False)
    elif suf == ".csv":
        df.to_csv(out_path, index=False)
    elif suf in (".xlsx", ".xls"):
        df.to_excel(out_path, index=False)
    else:
        raise ValueError(f"Unsupported output type: {suf}")


def make_train_val_hash(
    manifest_path: str,
    out_path: str,
    group_col: str | None = "group_id",
    val_frac: float = 0.2,
    preview: bool = True,
) -> pd.DataFrame:
    p = Path(manifest_path)
    df = _load_table(p)

    # Derive grouping key if missing
    if not group_col or group_col not in df.columns:
        if "output_tif" not in df.columns:
            raise ValueError(
                "group_col not found and 'output_tif' missing — cannot derive groups. "
                "Provide --group-col or include 'output_tif' in the table."
            )
        df["group_id"] = df["output_tif"].astype(str).apply(lambda s: str(Path(s).parent))
        group_col = "group_id"

    # Build deterministic uniform in [0,1) per group
    groups = df[group_col].astype(str).unique()
    g2r = {g: (stable_hash(g) % 10_000_000) / 10_000_000.0 for g in groups}

    # Assign split by group
    df = df.copy()
    df["split"] = df[group_col].map(lambda g: "val" if g2r[g] < val_frac else "train")

    if preview:
        print("Counts by split:")
        print(df["split"].value_counts(dropna=False))
        if "ds_factor" in df.columns:
            print("\nCounts by split × ds_factor:")
            print(df.groupby(["split", "ds_factor"]).size())

    out_p = Path(out_path)
    _save_table(df, out_p)
    print(f"\nSaved split table → {out_p}")
    return df


def main():
    ap = argparse.ArgumentParser(
        description="Deterministic, group-aware train/val split via stable hashing."
    )
    ap.add_argument("--manifest", required=True, help="Input table (.parquet/.csv/.xlsx)")
    ap.add_argument("--out", required=True, help="Output with split column (.parquet/.csv/.xlsx)")
    ap.add_argument("--group-col", default="group_id",
                    help="Grouping column; if missing, derived from parent of output_tif")
    ap.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction (0–1)")
    ap.add_argument("--no-preview", action="store_true", help="Disable printing split summaries")
    args = ap.parse_args()

    make_train_val_hash(
        manifest_path=args.manifest,
        out_path=args.out,
        group_col=args.group_col if args.group_col else None,
        val_frac=float(args.val_frac),
        preview=(not args.no_preview),
    )


if __name__ == "__main__":
    main()

## Example CLI Usage

# basic
#python split_train_val_hash.py `
#  --manifest "path\to\index.xlsx" `
#  --out "output\dir\index_with_split.parquet"

# with explicit grouping column and 25% val
#python split_train_val_hash.py `
#  --manifest "path\to\index.xlsx" `
#  --out "output\dir\index_with_split.parquet" `
#  --group-col "project_path" `
#  --val-frac 0.25