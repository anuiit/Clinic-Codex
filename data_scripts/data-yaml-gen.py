#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import yaml

def main():
    p = argparse.ArgumentParser(description="Génère un data.yaml pour Ultralytics à partir du dossier elements/")
    p.add_argument("--elements", default="elements", help="Dossier contenant 1 sous-dossier par classe (défaut: elements)")
    p.add_argument("--out", default="synthetic_dataset/data.yaml", help="Chemin de sortie du YAML (défaut: synthetic_dataset/data.yaml)")
    p.add_argument("--dataset-path", default="synthetic_dataset", help="Chemin racine du dataset (défaut: synthetic_dataset)")
    p.add_argument("--train", default="images", help="Chemin train relatif à dataset-path (défaut: images)")
    p.add_argument("--val", default="images", help="Chemin val relatif à dataset-path (défaut: images)")
    args = p.parse_args()

    elements_dir = Path(args.elements)
    if not elements_dir.exists():
        raise SystemExit(f"[ERR] Dossier introuvable: {elements_dir.resolve()}")

    # Récupérer les classes = noms des sous-dossiers (triés alphabétiquement, comme le générateur)
    class_names = sorted([p.name for p in elements_dir.iterdir() if p.is_dir()])
    if not class_names:
        raise SystemExit(f"[ERR] Aucun sous-dossier de classe trouvé dans: {elements_dir.resolve()}")

    data = {
        "path": args.dataset_path,
        "train": args.train,
        "val": args.val,
        "nc": len(class_names),
        "names": class_names
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print("[OK] data.yaml généré.")
    print(f" - Classes ({len(class_names)}): {class_names}")
    print(f" - Fichier : {out_path.resolve()}")
    print(f" - path/train/val : {data['path']} / {data['train']} / {data['val']}")

if __name__ == "__main__":
    main()
    