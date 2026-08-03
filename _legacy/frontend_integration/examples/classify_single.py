"""
Classify a single image using the CodexClassifier.

Usage:
    python classify_single.py path/to/image.jpg
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codex_model import CodexClassifier
from PIL import Image


def main():
    if len(sys.argv) < 2:
        print("Usage: python classify_single.py <image_path>")
        sys.exit(1)

    clf = CodexClassifier()
    img = Image.open(sys.argv[1]).convert("RGB")
    result = clf.classify(img)

    print(f"Element:    {result['class_name']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Rejected:   {result['rejected']}")
    for match in result["top_k"]:
        print(f"  {match['class_name']:<25} {match['confidence']:.3f}")


if __name__ == "__main__":
    main()
