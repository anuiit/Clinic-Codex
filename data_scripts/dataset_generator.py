#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset synthétique pour détection d'éléments dans des glyphes/pages.
- Prend plusieurs images par élément (1 dossier = 1 classe) depuis ./elements
- Compose des images synthétiques avec 2..K éléments aléatoires
- Augmentations réalistes (rotation, scale, jitter contraste/luminosité, blur léger, bruit papier)
- Exporte annotations en YOLO ou COCO
"""

import os, sys, json, math, uuid, random, glob
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# ---------------------- CONFIG ----------------------

ELEMENTS_DIR = "elements"          # dossiers par classe
OUTPUT_DIR   = "synthetic_dataset" # où écrire images & annotations
OUT_FORMAT   = "yolo"              # "yolo" ou "coco"
N_IMAGES     = 800                 # nombre d'images à générer
IMG_SIZE     = (1024, 1024)        # taille canevas (W,H)
MIN_PER_IMG  = 2                   # nb min d'instances par image
MAX_PER_IMG  = 6                   # nb max d'instances par image

# Augmentations
ROTATION_DEG = (-35, 35)
SCALE_RANGE  = (0.6, 1.3)          # facteur de mise à l’échelle
CONTRAST_JIT = (0.85, 1.15)
BRIGHT_JIT   = (0.9, 1.1)
BLUR_PROB    = 0.35
BLUR_RADIUS  = (0.6, 1.6)
ADD_PAPER_TEXTURE = True

# Pour COCO
COCO_LICENSES = [{"id": 1, "name": "unknown", "url": ""}]
COCO_INFO = {"year": 2025, "version": "1.0", "description": "synthetic glyph dataset", "contributor": "", "date_created": ""}

# ----------------------------------------------------

@dataclass
class PlacedInstance:
    class_id: int
    bbox_xyxy: Tuple[int, int, int, int]  # (x1,y1,x2,y2)

def ensure_dirs():
    if OUT_FORMAT.lower() == "yolo":
        Path(f"{OUTPUT_DIR}/images").mkdir(parents=True, exist_ok=True)
        Path(f"{OUTPUT_DIR}/labels").mkdir(parents=True, exist_ok=True)
    else:
        Path(f"{OUTPUT_DIR}/images").mkdir(parents=True, exist_ok=True)
        Path(f"{OUTPUT_DIR}").mkdir(parents=True, exist_ok=True)

def load_classes() -> Tuple[Dict[str,int], Dict[int,str], Dict[int, List[Path]]]:
    class_names = sorted([p.name for p in Path(ELEMENTS_DIR).iterdir() if p.is_dir()])
    if not class_names:
        print(f"[ERR] Aucun dossier de classe trouvé dans ./{ELEMENTS_DIR}")
        sys.exit(1)
    name2id = {c:i for i,c in enumerate(class_names)}
    id2name = {i:c for c,i in name2id.items()}

    samples: Dict[int, List[Path]] = {}
    for cname, cid in name2id.items():
        files = []
        for ext in ("*.png","*.jpg","*.jpeg","*.webp","*.bmp"):
            files += glob.glob(str(Path(ELEMENTS_DIR)/cname/ext))
        if not files:
            print(f"[WARN] Aucune image trouvée pour la classe '{cname}'")
        samples[cid] = [Path(f) for f in files]
    return name2id, id2name, samples

def jitter_image(img: Image.Image) -> Image.Image:
    # contraste / luminosité
    c = random.uniform(*CONTRAST_JIT)
    b = random.uniform(*BRIGHT_JIT)
    img = ImageEnhance.Contrast(img).enhance(c)
    img = ImageEnhance.Brightness(img).enhance(b)

    # blur aléatoire
    if random.random() < BLUR_PROB:
        r = random.uniform(*BLUR_RADIUS)
        img = img.filter(ImageFilter.GaussianBlur(radius=r))
    return img

def paper_texture(size: Tuple[int,int]) -> Image.Image:
    w,h = size
    base = Image.new("L", size, 235)  # papier clair
    arr = np.array(base, dtype=np.float32)
    # bruit doux
    noise = np.random.normal(0, 6, (h,w))
    arr = np.clip(arr + noise, 0, 255)
    # vignettage léger
    yy, xx = np.mgrid[:h,:w]
    cx, cy = w/2, h/2
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    arr -= 0.03 * dist
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")

def paste_with_bbox(canvas: Image.Image, elem: Image.Image, at_xy: Tuple[int,int]) -> Tuple[Image.Image, Tuple[int,int,int,int]]:
    """
    Colle elem (RGBA) sur canvas (RGB) en (x,y) . Retourne bbox xyxy de la zone non-transparente.
    """
    x,y = at_xy
    # s'assurer RGBA
    if elem.mode != "RGBA":
        elem = elem.convert("RGBA")

    tmp = Image.new("RGBA", canvas.size, (0,0,0,0))
    tmp.paste(elem, (x,y), elem)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), tmp).convert("RGB")

    # bbox via alpha
    alpha = np.array(tmp.split()[-1])
    ys, xs = np.where(alpha > 0)
    if xs.size == 0 or ys.size == 0:
        return canvas, (0,0,0,0)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return canvas, (int(x1), int(y1), int(x2), int(y2))

def place_random_instance(canvas: Image.Image, sample_path: Path) -> Image.Image:
    img = Image.open(sample_path).convert("RGBA")
    # binariser vite fait si fond blanc sale
    # img = ImageOps.autocontrast(img)

    # scale
    s = random.uniform(*SCALE_RANGE)
    new_w = max(8, int(img.width * s))
    new_h = max(8, int(img.height * s))
    img = img.resize((new_w, new_h), resample=Image.BICUBIC)

    # rotation (expand=True pour conserver l'objet complet)
    angle = random.uniform(*ROTATION_DEG)
    img = img.rotate(angle, expand=True, resample=Image.BICUBIC)

    # jitter
    img = jitter_image(img)

    # position aléatoire sans sortir du canevas
    W,H = canvas.size
    x = random.randint(0, max(0, W - img.width))
    y = random.randint(0, max(0, H - img.height))

    return img, (x,y)

def yolo_line(class_id: int, bbox_xyxy: Tuple[int,int,int,int], img_size: Tuple[int,int]) -> str:
    x1,y1,x2,y2 = bbox_xyxy
    W,H = img_size
    # clamp
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(W-1,x2), min(H-1,y2)
    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)
    cx = x1 + bw/2.0
    cy = y1 + bh/2.0
    return f"{class_id} {cx/W:.6f} {cy/H:.6f} {bw/W:.6f} {bh/H:.6f}"

def main():
    random.seed(42)
    ensure_dirs()
    name2id, id2name, samples = load_classes()

    # Vérif : au moins une image par classe
    for cid, files in samples.items():
        if not files:
            print(f"[WARN] Classe vide: {id2name[cid]} (pas d'exemples)")

    coco = {
        "info": COCO_INFO,
        "licenses": COCO_LICENSES,
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": n, "supercategory": "element"} for n,i in name2id.items()]
    }
    ann_id = 1

    for i in range(N_IMAGES):
        # fond
        if ADD_PAPER_TEXTURE:
            canvas = paper_texture(IMG_SIZE)
        else:
            canvas = Image.new("RGB", IMG_SIZE, (245,245,245))

        instances: List[PlacedInstance] = []

        # combien d'instances sur cette image ?
        k = random.randint(MIN_PER_IMG, MAX_PER_IMG)

        # pour chaque instance, choisir une classe puis un exemplaire
        for _ in range(k):
            # choisir une classe non vide
            non_empty = [cid for cid, lst in samples.items() if len(lst) > 0]
            if not non_empty:
                print("[ERR] Aucune image d'élément disponible.")
                sys.exit(1)
            cid = random.choice(non_empty)
            spath = random.choice(samples[cid])

            elem_img, (x,y) = place_random_instance(canvas, spath)
            canvas, bbox = paste_with_bbox(canvas, elem_img, (x,y))

            x1,y1,x2,y2 = bbox
            if (x2-x1) < 3 or (y2-y1) < 3:
                # trop petit / invalide, on saute
                continue
            instances.append(PlacedInstance(class_id=cid, bbox_xyxy=bbox))

        # enregistrer
        img_id = i + 1
        fname = f"{img_id:06d}.jpg"
        canvas.save(f"{OUTPUT_DIR}/images/{fname}", quality=92)

        if OUT_FORMAT.lower() == "yolo":
            # un .txt par image
            lbl_path = Path(f"{OUTPUT_DIR}/labels/{Path(fname).stem}.txt")
            lines = [yolo_line(inst.class_id, inst.bbox_xyxy, canvas.size) for inst in instances]
            lbl_path.write_text("\n".join(lines), encoding="utf-8")
        else:
            # COCO : accumulate
            coco["images"].append({
                "id": img_id,
                "file_name": f"images/{fname}",
                "width": canvas.size[0],
                "height": canvas.size[1],
                "license": 1
            })
            for inst in instances:
                x1,y1,x2,y2 = inst.bbox_xyxy
                w, h = x2 - x1 + 1, y2 - y1 + 1
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": inst.class_id,
                    "bbox": [int(x1), int(y1), int(w), int(h)],
                    "area": int(w*h),
                    "iscrowd": 0
                })
                ann_id += 1

        if (i+1) % 50 == 0:
            print(f"[OK] {i+1}/{N_IMAGES} images générées")

    if OUT_FORMAT.lower() == "coco":
        with open(f"{OUTPUT_DIR}/annotations_coco.json", "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False, indent=2)
        print(f"[DONE] COCO JSON -> {OUTPUT_DIR}/annotations_coco.json")

    print("[DONE] Génération terminée.")
    print(f"Format: {OUT_FORMAT.upper()}  |  Images: {N_IMAGES}  |  Taille: {IMG_SIZE}")

if __name__ == "__main__":
    main()
