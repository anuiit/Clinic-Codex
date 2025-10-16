#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, re
from pathlib import Path
import yaml

def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_label_file(p):
    lines = []
    for line in p.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        parts = re.split(r"\s+", line.strip())
        if len(parts) != 5:  # class cx cy w h (normalized)
            raise ValueError(f"{p}: bad line -> {line}")
        cls, cx, cy, w, h = parts
        lines.append((int(cls), float(cx), float(cy), float(w), float(h)))
    return lines

def main():
    data_yaml = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_dataset/data.yaml")
    data = load_yaml(data_yaml)
    root = Path(data["path"])
    img_dir = root / data["train"]  # si train==images (set unique)
    lbl_dir = root / "labels"

    if not img_dir.exists() or not lbl_dir.exists():
        sys.exit(f"[ERR] Dossiers introuvables: {img_dir} / {lbl_dir}")

    nc = int(data["nc"])
    names = data["names"]
    if nc != len(names):
        print(f"[WARN] nc({nc}) != len(names)({len(names)})")

    # map images->labels
    exts = {".jpg",".jpeg",".png",".bmp",".webp"}
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
    if not images:
        sys.exit(f"[ERR] Aucune image dans: {img_dir}")

    bad_missing_lbl = []
    bad_class = []
    bad_bbox = []
    class_count = [0]*nc

    for im in images:
        lab = lbl_dir / f"{im.stem}.txt"
        if not lab.exists():
            bad_missing_lbl.append(im.name)
            continue
        try:
            rows = read_label_file(lab)
        except Exception as e:
            sys.exit(f"[ERR] {lab}: {e}")

        for (cls, cx, cy, w, h) in rows:
            if cls < 0 or cls >= nc:
                bad_class.append((lab.name, cls))
            # bboxes normalisées dans (0,1], tolérance
            for val in (cx, cy, w, h):
                if not (0.0 < val <= 1.0):
                    bad_bbox.append((lab.name, (cx,cy,w,h)))
                    break
            if 0 <= cls < nc:
                class_count[cls] += 1

    # Résultats
    print("\n[CHECK] Dataset:", root)
    print(f"Images: {len(images)}")
    print("Classes:", names)
    print("Instances par classe:")
    for i, n in enumerate(names):
        print(f"  - {i} ({n}): {class_count[i]}")

    if bad_missing_lbl:
        print(f"\n[WARN] Labels manquants ({len(bad_missing_lbl)}):")
        for x in bad_missing_lbl[:10]:
            print("  ", x)
        if len(bad_missing_lbl) > 10:
            print("  ...")

    if bad_class:
        print(f"\n[ERR] IDs de classe hors-plage ({len(bad_class)}) (attendu 0..{nc-1}):")
        for fn, c in bad_class[:10]:
            print(f"  {fn}: {c}")
        if len(bad_class) > 10:
            print("  ...")

    if bad_bbox:
        print(f"\n[ERR] Bboxes non normalisées ou invalides ({len(bad_bbox)}):")
        for fn, bb in bad_bbox[:10]:
            print(f"  {fn}: {bb}")
        if len(bad_bbox) > 10:
            print("  ...")

    if not bad_missing_lbl and not bad_class and not bad_bbox:
        print("\n[OK] Tout est cohérent ✅")

if __name__ == "__main__":
    main()
