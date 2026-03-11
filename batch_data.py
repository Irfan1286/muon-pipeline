"""
batch_data.py
─────────────
Splits  muon-pipeline/data/dummy_data.csv  into batch files of 10,000 rows
each, writing them to  muon-pipeline/data/batches/.

Output files are named:
    batch_0001.csv   (rows      1 – 10 000)
    batch_0002.csv   (rows 10 001 – 20 000)
    …

Usage:
    python batch_data.py                      # uses defaults below
    python batch_data.py --batch-size 5000    # custom batch size
    python batch_data.py --help
"""

import argparse
import math
import os
import sys

import pandas as pd


# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_INPUT  = os.path.join(os.path.dirname(__file__), "data", "dummy_data.csv")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "data", "batches")
DEFAULT_BATCH_SIZE = 10_000


# ── Helpers ─────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch dummy_data.csv into fixed-size CSV chunks."
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        help=f"Path to source CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=DEFAULT_OUTPUT,
        help=f"Directory to write batch CSVs (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of rows per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    return parser.parse_args()


def batch_csv(input_path: str, output_dir: str, batch_size: int) -> None:
    """Read *input_path*, split into *batch_size*-row chunks, write to *output_dir*."""

    # ── Validate input ───────────────────────────────────────────────────────
    if not os.path.isfile(input_path):
        sys.exit(f"[ERROR] Input file not found: {input_path}")

    if batch_size < 1:
        sys.exit(f"[ERROR] --batch-size must be a positive integer, got {batch_size}")

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"[INFO]  Reading: {input_path}")
    df = pd.read_csv(input_path)
    total_rows  = len(df)
    total_batches = math.ceil(total_rows / batch_size)

    print(f"[INFO]  Total rows    : {total_rows:,}")
    print(f"[INFO]  Batch size    : {batch_size:,}")
    print(f"[INFO]  Total batches : {total_batches}")

    # ── Prepare output directory ─────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO]  Output dir    : {output_dir}\n")

    # ── Write batches ────────────────────────────────────────────────────────
    pad_width = len(str(total_batches))          # e.g. 2 → "01", 10 → "001"

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end   = min(start + batch_size, total_rows)

        chunk     = df.iloc[start:end]
        batch_num = str(batch_idx + 1).zfill(max(pad_width, 4))
        filename  = f"batch_{batch_num}.csv"
        out_path  = os.path.join(output_dir, filename)

        chunk.to_csv(out_path, index=False)
        print(f"  [+] {filename}  ({len(chunk):,} rows  |  rows {start+1:,}–{end:,})")

    print(f"\n[DONE]  {total_batches} batch file(s) written to '{output_dir}'")


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    batch_csv(
        input_path=args.input,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
