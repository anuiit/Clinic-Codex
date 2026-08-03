#!/usr/bin/env python3
"""
Build metadata CSV from the Elements folder structure.

Usage:
    python -m codex_pipeline.scripts.build_metadata
    python -m codex_pipeline.scripts.build_metadata --elements-dir ./Elements --output metadata.csv
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from codex_pipeline.data.metadata import build_metadata, get_class_stats


def main():
    parser = argparse.ArgumentParser(description="Build metadata CSV from Elements/")
    parser.add_argument(
        "--elements-dir",
        type=str,
        default="./Elements",
        help="Path to Elements directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./metadata.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    elements_dir = Path(args.elements_dir).resolve()
    if not elements_dir.exists():
        print(f"Error: {elements_dir} does not exist")
        sys.exit(1)

    print(f"Scanning {elements_dir}...")
    df = build_metadata(str(elements_dir))

    # Save
    output_path = Path(args.output).resolve()
    df.to_csv(output_path, index=False)
    print(f"Saved metadata to {output_path}")

    # Print summary
    stats = get_class_stats(df)
    print(f"\n{'='*60}")
    print(f"Dataset Summary")
    print(f"{'='*60}")
    print(f"Total images:  {len(df)}")
    print(f"Total classes:  {df['class_label'].nunique()}")
    print(f"Median images/class: {stats['count'].median():.0f}")
    print(f"Mean images/class:   {stats['count'].mean():.1f}")
    print(f"Min images/class:    {stats['count'].min()}")
    print(f"Max images/class:    {stats['count'].max()}")

    # Distribution
    bins = [(1, 1), (2, 3), (4, 5), (6, 10), (11, 20), (21, 50)]
    print(f"\nClass size distribution:")
    for lo, hi in bins:
        count = ((stats["count"] >= lo) & (stats["count"] <= hi)).sum()
        print(f"  {lo:>3}–{hi:<3} images: {count:>3} classes")

    # Top and bottom classes
    print(f"\nTop 10 classes:")
    for _, row in stats.head(10).iterrows():
        print(f"  {row['element_name']:<30} {int(row['count']):>3} images")

    print(f"\nBottom 10 classes:")
    for _, row in stats.tail(10).iterrows():
        print(f"  {row['element_name']:<30} {int(row['count']):>3} images")


if __name__ == "__main__":
    main()
