"""
Build and load metadata CSV from the Elements folder structure.

Folder naming: NNNN-element_name (e.g., 0015-cacahuatl)
File naming:   CC_FF_PP-ID.bmp  (codex_folio_page-elementID)
"""

import os
import re
import pandas as pd
from pathlib import Path


def parse_folder_name(folder_name: str) -> dict:
    """Parse 'NNNN-element_name' into element_id and element_name."""
    match = re.match(r"^(\d+)-(.+)$", folder_name)
    if not match:
        return None
    return {
        "element_id": int(match.group(1)),
        "element_name": match.group(2),
    }


def parse_filename(filename: str) -> dict:
    """Parse 'CC_FF_PP-ID.bmp' into codex, folio, page, instance_id."""
    stem = Path(filename).stem
    match = re.match(r"^(\d+)_(\d+)_(\d+)-(.+)$", stem)
    if not match:
        return {"codex": None, "folio": None, "page": None, "instance_id": stem}
    return {
        "codex": match.group(1),
        "folio": match.group(2),
        "page": match.group(3),
        "instance_id": match.group(4),
    }


def build_metadata(
    elements_dir: str,
    read_image_sizes: bool = False,
    base_dir: str = None,
) -> pd.DataFrame:
    """
    Scan Elements directory and build a metadata DataFrame.

    Args:
        elements_dir: path to Elements/ directory
        read_image_sizes: if True, open each image to read width/height (slow)
        base_dir: root directory used to compute relative image paths.
                  Defaults to the parent of elements_dir (i.e., project root).

    Returns DataFrame with columns:
        image_path, element_id, element_name, class_label,
        codex, folio, page, instance_id [, width, height]
    """
    records = []
    elements_dir = Path(elements_dir).resolve()

    # Determine project root for relative path computation
    project_root = Path(base_dir).resolve() if base_dir is not None else elements_dir.parent

    folders = sorted(
        [f for f in elements_dir.iterdir() if f.is_dir()],
        key=lambda f: f.name,
    )

    class_label = 0
    class_map = {}

    for folder in folders:
        parsed = parse_folder_name(folder.name)
        if parsed is None:
            continue

        bmp_files = sorted(folder.glob("*.bmp"))
        if not bmp_files:
            continue

        key = (parsed["element_id"], parsed["element_name"])
        if key not in class_map:
            class_map[key] = class_label
            class_label += 1

        for img_path in bmp_files:
            file_info = parse_filename(img_path.name)

            record = {
                "image_path": str(os.path.relpath(img_path, project_root)),
                "element_id": parsed["element_id"],
                "element_name": parsed["element_name"],
                "class_label": class_map[key],
                "codex": file_info["codex"],
                "folio": file_info["folio"],
                "page": file_info["page"],
                "instance_id": file_info["instance_id"],
            }

            if read_image_sizes:
                from PIL import Image
                try:
                    with Image.open(img_path) as img:
                        record["width"], record["height"] = img.size
                except Exception:
                    record["width"], record["height"] = None, None

            records.append(record)

    df = pd.DataFrame(records)
    return df


def load_metadata(csv_path: str) -> pd.DataFrame:
    """Load metadata CSV."""
    return pd.read_csv(csv_path)


def get_class_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-class statistics."""
    agg_dict = {"count": ("image_path", "count")}

    if "width" in df.columns:
        agg_dict["avg_width"] = ("width", "mean")
    if "height" in df.columns:
        agg_dict["avg_height"] = ("height", "mean")

    stats = (
        df.groupby(["class_label", "element_id", "element_name"])
        .agg(**agg_dict)
        .reset_index()
        .sort_values("count", ascending=False)
    )
    return stats


def filter_classes(df: pd.DataFrame, min_images: int = 2) -> pd.DataFrame:
    """Filter out classes with fewer than min_images examples."""
    counts = df["class_label"].value_counts()
    valid_classes = counts[counts >= min_images].index
    return df[df["class_label"].isin(valid_classes)].copy()
