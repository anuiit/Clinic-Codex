"""
Classify all images in a folder.

Usage:
    python classify_batch.py path/to/folder/
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codex_model import CodexClassifier

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}


def main():
    if len(sys.argv) < 2:
        print("Usage: python classify_batch.py <folder>")
        sys.exit(1)

    folder = Path(sys.argv[1])
    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED)

    clf = CodexClassifier()
    images = [np.array(Image.open(p).convert("RGB")) for p in paths]

    results = clf.classify_batch(images)
    for path, result in zip(paths, results):
        status = "REJECTED" if result["rejected"] else result["class_name"]
        print(f"{path.name:<40} -> {status:<20} ({result['confidence']:.3f})")


if __name__ == "__main__":
    main()
