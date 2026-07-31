#!/usr/bin/env python3
"""Write the current runtime class order as a hashable baseline artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.codex_pipeline.data.class_order import class_order_sha256, load_runtime_class_order


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, default=ROOT / "backend/codex_model/config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "backend/codex_pipeline/config/class_order.json")
    args = parser.parse_args()

    class_names = load_runtime_class_order(args.runtime_config)
    payload = {
        "schema_version": "baseline-class-order.v1",
        "source_config": str(args.runtime_config.resolve()),
        "class_count": len(class_names),
        "class_names": class_names,
        "sha256": class_order_sha256(class_names),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(class_names)} classes, sha256={payload['sha256']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
