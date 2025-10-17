#!/usr/bin/env python
# convert_manifest_to_excel.py
import os
import sys
from pathlib import Path
from typing import Optional, List
import pandas as pd


def convert_manifest_to_excel(
    manifest_path: str | os.PathLike,
    out_xlsx: str | os.PathLike,
    filter_query: str | None = None,
    select_cols: list[str] | None = None,
    sort_by: list[str] | None = None,
) -> Path:
    """
    Convert a deblur3d manifest (parquet/csv) to an Excel workbook.

    - filter_query: optional pandas query string, e.g. "n_slices>=128 and size_8bit_GB<6"
    - select_cols: optional list of columns to keep (in this order)
    - sort_by:     optional list of columns to sort by (ascending)
    """
    mp = Path(manifest_path)
    if not mp.exists():
        raise FileNotFoundError(f"Manifest not found: {mp}")

    # Load
    if mp.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(mp)
        except Exception as e:
            raise RuntimeError(
                "Failed to read Parquet. Ensure 'pyarrow' is installed (pip install pyarrow)."
            ) from e
    else:
        df = pd.read_csv(mp)

    # Filter
    if filter_query:
        try:
            df = df.query(filter_query, engine="python").copy()
        except Exception as e:
            raise ValueError(f"Invalid filter_query: {filter_query}\n{e}")

    if df.empty:
        raise RuntimeError("No rows after filtering; nothing to write.")

    # Derived sizes for readability (if not already present)
    if {"n_slices", "H", "W"}.issubset(df.columns) and "size_8bit_GB" not in df.columns:
        df["size_8bit_GB"] = (df["n_slices"].astype("float64") * df["H"] * df["W"]) / 1e9

    if "xml_bytes_8bit_from_xml" in df.columns and "size_8bit_GB_xml" not in df.columns:
        df["size_8bit_GB_xml"] = df["xml_bytes_8bit_from_xml"] / 1e9

    # Column selection
    if select_cols:
        keep = [c for c in select_cols if c in df.columns]
        if not keep:
            raise ValueError("None of the requested select_cols are in the manifest.")
        df = df[keep]

    # Sorting
    if sort_by:
        sort_cols = [c for c in sort_by if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=True, kind="mergesort").reset_index(drop=True)

    # Summary sheet
    summary = {
        "rows": [len(df)],
        "unique_projects": [df["project_path"].nunique() if "project_path" in df.columns else None],
        "total_size_8bit_GB": [df["size_8bit_GB"].sum() if "size_8bit_GB" in df.columns else None],
        "total_size_8bit_GB_xml": [df["size_8bit_GB_xml"].sum() if "size_8bit_GB_xml" in df.columns else None],
        "mean_H": [df["H"].mean() if "H" in df.columns else None],
        "mean_W": [df["W"].mean() if "W" in df.columns else None],
        "mean_n_slices": [df["n_slices"].mean() if "n_slices" in df.columns else None],
    }
    df_summary = pd.DataFrame(summary)

    # Write Excel
    out = Path(out_xlsx)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Scans", index=False)
            df_summary.to_excel(writer, sheet_name="Summary", index=False)

            wb = writer.book
            ws1 = writer.sheets["Scans"]
            ws2 = writer.sheets["Summary"]

            # Tables
            if len(df) > 0:
                ws1.add_table(
                    0, 0, len(df), max(0, len(df.columns) - 1),
                    {"name": "ScansTable", "columns": [{"header": c} for c in df.columns]}
                )
            ws2.add_table(
                0, 0, len(df_summary), max(0, len(df_summary.columns) - 1),
                {"name": "SummaryTable", "columns": [{"header": c} for c in df_summary.columns]}
            )

            # Column widths (quick autofit)
            for i, c in enumerate(df.columns):
                try:
                    width = min(60, max(10, int(df[c].astype(str).str.len().quantile(0.95)) + 2))
                except Exception:
                    width = max(10, len(str(c)) + 2)
                ws1.set_column(i, i, width)
            for i, c in enumerate(df_summary.columns):
                ws2.set_column(i, i, max(12, len(c) + 2))
    except Exception as e:
        raise RuntimeError(
            "Failed to write Excel. Ensure 'xlsxwriter' is installed (pip install XlsxWriter)."
        ) from e

    print(f"Saved Excel → {out}")
    return out


# ---------------- CLI ----------------
def _split_arg_list(v: Optional[str]) -> Optional[List[str]]:
    """Split comma/semicolon-separated lists; allow None."""
    if v is None:
        return None
    # support comma or semicolon, strip spaces
    parts = [p.strip() for p in v.replace(";", ",").split(",")]
    return [p for p in parts if p]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Convert a deblur3d manifest (parquet/csv) to an Excel workbook.")
    ap.add_argument("--manifest", required=True, help="Path to manifest (.parquet or .csv)")
    ap.add_argument("--out", required=True, help="Path to output .xlsx file")
    ap.add_argument("--filter", default=None, help="Optional pandas query, e.g. \"n_slices>=128 and size_8bit_GB<6\"")
    ap.add_argument("--select-cols", default=None,
                    help="Comma/semicolon-separated column names to keep (in order)")
    ap.add_argument("--sort-by", default=None,
                    help="Comma/semicolon-separated column names to sort by (ascending)")
    args = ap.parse_args()

    select_cols = _split_arg_list(args.select_cols)
    sort_by = _split_arg_list(args.sort_by)

    convert_manifest_to_excel(
        manifest_path=args.manifest,
        out_xlsx=args.out,
        filter_query=args.filter,
        select_cols=select_cols,
        sort_by=sort_by,
    )


if __name__ == "__main__":
    main()
