"""
Export trained prototypes into the standalone codex_model/ package.

Usage:
    python -m codex_pipeline.scripts.export_model

What it does:
    1. Loads prototypes/prototypes.pt (full training artefact)
    2. Writes codex_model/weights/prototypes.pt  — prototypes tensor + class info
    3. Writes codex_model/weights/projection.pt  — projection head state_dict only
    4. Writes codex_model/config.json            — updates num_classes + class_names
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


def main() -> None:
    # ------------------------------------------------------------------ paths
    project_root = Path(__file__).resolve().parents[2]  # CODEX_New/
    src_path = project_root / "prototypes" / "prototypes.pt"
    weights_dir = project_root / "codex_model" / "weights"
    config_path = project_root / "codex_model" / "config.json"

    weights_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------- load full artefact
    print(f"Loading: {src_path}")
    if not src_path.exists():
        print(f"ERROR: {src_path} not found. Train the model first.", file=sys.stderr)
        sys.exit(1)

    data = torch.load(src_path, map_location="cpu", weights_only=False)

    required_keys = {"prototypes", "class_names", "class_labels",
                     "embedding_dim", "hidden_dim", "model_state_dict"}
    missing = required_keys - set(data.keys())
    if missing:
        print(f"ERROR: prototypes.pt is missing keys: {missing}", file=sys.stderr)
        sys.exit(1)

    # -------------------------------------------------------- split and save
    # 1) Prototypes file — everything except the model weights
    proto_out = {
        "prototypes":   data["prototypes"],
        "class_names":  data["class_names"],   # {int: str}
        "class_labels": data["class_labels"],
        "embedding_dim": data["embedding_dim"],
    }
    proto_dest = weights_dir / "prototypes.pt"
    torch.save(proto_out, proto_dest)
    print(f"Saved prototypes  → {proto_dest}  "
          f"(shape {data['prototypes'].shape})")

    # 2) Projection head weights only
    proj_dest = weights_dir / "projection.pt"
    torch.save(data["model_state_dict"], proj_dest)
    print(f"Saved projection  → {proj_dest}  "
          f"({len(data['model_state_dict'])} tensors)")

    # -------------------------------------------------------- update config.json
    class_names_dict: dict = data["class_names"]      # {int: str}
    num_classes = len(class_names_dict)
    # Build ordered list aligned to sorted class labels.
    # class_names keys are class_label integers (can be sparse / not 0..N-1).
    sorted_labels = sorted(class_names_dict.keys())
    class_names_list = [class_names_dict[lbl] for lbl in sorted_labels]

    with open(config_path, "r") as f:
        config = json.load(f)

    config["num_classes"] = num_classes
    config["class_names"] = class_names_list
    config["embedding_dim"] = int(data["embedding_dim"])
    config["hidden_dim"] = int(data["hidden_dim"])

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Updated config    → {config_path}")

    # -------------------------------------------------------- summary
    print()
    print("=" * 60)
    print(f"  Export complete")
    print(f"  Classes  : {num_classes}")
    print(f"  Proto dim: {data['prototypes'].shape[1]}")
    print(f"  Examples : {class_names_list[:5]}{'...' if num_classes > 5 else ''}")
    print("=" * 60)
    print()
    print("Next: from codex_model import CodexClassifier; clf = CodexClassifier()")


if __name__ == "__main__":
    main()
