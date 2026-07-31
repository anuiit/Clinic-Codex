"""Canonical runtime class-order helpers.

The integer label order is a runtime compatibility contract.  It comes from
``backend/codex_model/config.json`` and must never be replaced by alphabetical
sorting of display names.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_runtime_class_order(config_path: str | Path) -> list[str]:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    class_names = payload.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError(f"runtime config has no class_names list: {config_path}")
    if not all(isinstance(name, str) and name for name in class_names):
        raise ValueError(f"runtime config has invalid class_names: {config_path}")
    if len(set(class_names)) != len(class_names):
        raise ValueError(f"runtime config has duplicate class_names: {config_path}")
    return class_names


def class_order_sha256(class_names: list[str]) -> str:
    encoded = json.dumps(class_names, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_metadata_class_order(metadata, class_names: list[str]) -> None:
    """Require a one-to-one, exact label-to-name runtime agreement.

    Labels originating in CSV files may be represented as ``0.0`` by pandas;
    integral numeric labels are accepted, while fractional or conflicting
    values are rejected instead of being silently coerced.
    """

    required_columns = {"class_label", "element_name"}
    missing = required_columns - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")

    label_to_name: dict[int, str] = {}
    for label, name in metadata[["class_label", "element_name"]].itertuples(index=False):
        try:
            normalized_label = int(label)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"metadata has non-integral class label: {label!r}") from exc
        if normalized_label != label:
            raise ValueError(f"metadata has non-integral class label: {label!r}")
        if not isinstance(name, str) or not name:
            raise ValueError(f"metadata has invalid element name for label {normalized_label}: {name!r}")
        previous = label_to_name.setdefault(normalized_label, name)
        if previous != name:
            raise ValueError(
                f"metadata maps class label {normalized_label} to conflicting names: "
                f"{previous!r} and {name!r}"
            )

    expected_labels = list(range(len(class_names)))
    labels = sorted(label_to_name)
    names = [label_to_name[label] for label in labels]
    if labels != expected_labels or names != class_names:
        raise ValueError(
            "metadata class order does not match the runtime contract; "
            f"expected {len(class_names)} ordered classes, got {len(names)}"
        )
