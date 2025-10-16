#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random, shutil
from pathlib import Path

ROOT = Path("synthetic_dataset")
IMG = ROOT / "images"
LBL = ROOT / "labels"
VAL_RATIO = 0.15
SEED = 42

def main():
    random.seed(SEED)
    exts = {".jpg",".jpeg",".png",".bmp",".webp"}
    files = sorted([p for p in IMG.iterdir() if p.suffix.lower() in exts])
    random.shuffle(files)

    n_val = int(len(files) * VAL_RATIO)
    val_set = set(f.stem for f in files[:n_val])

    # make dirs
    for sub in ["images/train","images/val","labels/train","labels/val"]:
        (ROOT / sub).mkdir(parents=True, exist_ok=True)

    for f in files:
        subset = "val" if f.stem in val_set else "train"
        shutil.copy2(f, ROOT / f"images/{subset}/{f.name}")
        lab = LBL / f"{f.stem}.txt"
        if lab.exists():
            shutil.copy2(lab, ROOT / f"labels/{subset}/{lab.name}")

    print(f"[OK] Split done: train={len(files)-n_val}  val={n_val}")

if __name__ == "__main__":
    main()
