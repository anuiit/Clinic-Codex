"""Benchmark the actual runtime and candidate classifier packages.

By default this loads the ``prototypes.pt`` stored in each package.  ``refit``
is deliberately opt-in and only useful as a projection-head diagnostic: it
rebuilds prototypes from the train split, so it must not be used as a
promotion-quality result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.codex_pipeline.data.class_order import load_runtime_class_order  # noqa: E402
from backend.codex_pipeline.models.projection_head import ProjectionHead  # noqa: E402
from backend.codex_model.classifier import PREPROCESSING_VERSION  # noqa: E402

BENCHMARK_SCHEMA_VERSION = "benchmark-model-replacement.v2"
TRAIN_SPLIT_NAMES = {"train"}
EVAL_SPLIT_PRIORITY = ("test", "val", "evaluation", "eval", "dev", "locked_test")
DEFAULT_BATCH_SIZE = 4096
STORED_PROTOTYPES = "stored"
REFIT_PROTOTYPES = "refit"


@dataclass(frozen=True)
class ModelPackage:
    root: Path
    config_path: Path
    weights_dir: Path
    projection_path: Path
    prototypes_path: Path
    class_names: list[str]
    class_labels: list[int]
    prototype_names: list[str]
    prototypes: torch.Tensor
    num_classes: int
    hidden_dim: int
    embedding_dim: int
    rejection_threshold: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_json_value(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_package_root(path: Path) -> Path:
    path = path.resolve()
    direct_config, direct_weights = path / "config.json", path / "weights"
    if direct_config.is_file() and (direct_weights / "projection.pt").is_file() and (direct_weights / "prototypes.pt").is_file():
        return path
    runtime = path / "runtime"
    if (runtime / "config.json").is_file() and (runtime / "weights" / "projection.pt").is_file() and (runtime / "weights" / "prototypes.pt").is_file():
        return runtime
    raise FileNotFoundError("package directory must contain config.json + weights/ or runtime/config.json + runtime/weights/ " f"(got {path})")


def _load_model_package(path: Path) -> ModelPackage:
    root = _resolve_package_root(path)
    config_path, weights_dir = root / "config.json", root / "weights"
    projection_path, prototypes_path = weights_dir / "projection.pt", weights_dir / "prototypes.pt"
    config = _load_json_value(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"expected JSON object: {config_path}")
    class_names = load_runtime_class_order(config_path)
    num_classes = int(config.get("num_classes", len(class_names)))
    if len(class_names) != num_classes:
        raise ValueError(f"runtime config class_names length does not match num_classes: {config_path}")
    data = torch.load(prototypes_path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError(f"prototype artifact must be a mapping: {prototypes_path}")
    prototypes, raw_labels, raw_names = data.get("prototypes"), data.get("class_labels"), data.get("class_names")
    if not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 2 or tuple(prototypes.shape) != (num_classes, int(config["embedding_dim"])):
        raise ValueError(f"prototype shape does not match config: {prototypes_path}")
    if not prototypes.is_floating_point() or not torch.isfinite(prototypes).all():
        raise ValueError(f"prototypes must be finite floating-point values: {prototypes_path}")
    if not isinstance(raw_labels, torch.Tensor) or raw_labels.ndim != 1 or raw_labels.numel() != num_classes or raw_labels.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        raise ValueError(f"prototype class_labels must be an integer vector matching num_classes: {prototypes_path}")
    class_labels = [int(value) for value in raw_labels.tolist()]
    if class_labels != sorted(class_labels) or len(set(class_labels)) != len(class_labels):
        raise ValueError(f"prototype class_labels must be unique and sorted: {prototypes_path}")
    if not isinstance(raw_names, dict) or set(raw_names) != set(class_labels):
        raise ValueError(f"prototype class_names and class_labels disagree: {prototypes_path}")
    if any(not isinstance(label, int) or isinstance(label, bool) or not isinstance(name, str) or not name for label, name in raw_names.items()):
        raise ValueError(f"prototype class_names mapping is invalid: {prototypes_path}")
    prototype_names = [raw_names[label] for label in class_labels]
    if prototype_names != class_names:
        raise ValueError(f"prototype taxonomy does not match config class_names: {prototypes_path}")
    return ModelPackage(root, config_path, weights_dir, projection_path, prototypes_path, class_names, class_labels, prototype_names, prototypes, num_classes, int(config["hidden_dim"]), int(config["embedding_dim"]), float(config.get("rejection_threshold", 0.35)))


def _normalise_split_name(value: Any) -> str:
    return str(value).strip().lower()


def _get_record_field(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    raise KeyError(f"split manifest row is missing required field(s): {', '.join(names)}")


def _load_split_records(path: Path) -> list[dict[str, Any]]:
    payload = _load_json_value(path)
    for key in ("rows", "samples", "entries", "items", "split_rows"):
        if isinstance(payload, dict) and isinstance(payload.get(key), list):
            payload = payload[key]
            break
    if not isinstance(payload, list) or not all(isinstance(record, dict) for record in payload):
        raise ValueError(f"unsupported split manifest schema: {path}")
    return payload


def _extract_row_index(record: dict[str, Any]) -> int | None:
    for key in ("index", "row_index", "feature_index"):
        if key in record:
            return int(record[key])
    return None


def _extract_split_name(record: dict[str, Any]) -> str:
    return _normalise_split_name(_get_record_field(record, "dataset_split", "split", "subset", "partition"))


def _extract_label(record: dict[str, Any]) -> int:
    return int(_get_record_field(record, "class_label", "label", "element_label"))


def _extract_class_name(record: dict[str, Any]) -> str:
    value = _get_record_field(record, "class_name", "element_name")
    if not isinstance(value, str) or not value:
        raise ValueError("split manifest has an invalid class_name")
    return value


def _extract_path_key(record: dict[str, Any]) -> str | None:
    for key in ("image_path", "output_path", "source_path", "path"):
        if key in record and record[key] is not None:
            return str(record[key])
    return None


def _resolve_indices(records: list[dict[str, Any]], *, features_len: int, image_paths: list[str] | None, row_ids: list[str] | None = None) -> list[int]:
    if row_ids is not None and all(record.get("row_id") for record in records):
        if len(row_ids) != features_len or len(set(row_ids)) != features_len:
            raise ValueError("cached row_ids must be unique and align with features")
        id_to_index = {row_id: index for index, row_id in enumerate(row_ids)}
        try:
            indices = [id_to_index[record["row_id"]] for record in records]
        except KeyError as exc:
            raise ValueError("snapshot row_id not found in feature cache") from exc
    elif all(_extract_row_index(record) is not None for record in records):
        indices = [int(_extract_row_index(record)) for record in records]
    elif image_paths is not None:
        path_to_index = {str(path): index for index, path in enumerate(image_paths)}
        try:
            indices = [path_to_index[_extract_path_key(record) or ""] for record in records]
        except KeyError as exc:
            raise ValueError("split manifest path not found in features payload") from exc
    else:
        if len(records) > features_len:
            raise ValueError("split manifest has more rows than cached features")
        indices = list(range(len(records)))
    if len(set(indices)) != len(indices):
        raise ValueError("split manifest maps multiple rows to the same cached feature index")
    if any(index < 0 or index >= features_len for index in indices):
        raise ValueError("cached feature index out of bounds")
    return indices


def _select_eval_split(records: list[dict[str, Any]]) -> str:
    splits = {_extract_split_name(record) for record in records}
    for candidate in EVAL_SPLIT_PRIORITY:
        if candidate in splits:
            return candidate
    non_train = sorted(split for split in splits if split not in TRAIN_SPLIT_NAMES)
    if non_train:
        return non_train[0]
    raise ValueError("split manifest contains train rows but no evaluation split")


def _validate_split_integrity(train_records: list[dict[str, Any]], eval_records: list[dict[str, Any]], train_indices: list[int], eval_indices: list[int], labels: torch.Tensor, class_names: list[str]) -> list[str]:
    overlap = sorted(set(train_indices) & set(eval_indices))
    if overlap:
        raise ValueError(f"train/evaluation cached feature indices overlap: {overlap[:10]}")
    for record, index in zip(train_records + eval_records, train_indices + eval_indices):
        dense_label, name = _extract_label(record), _extract_class_name(record)
        if not 0 <= dense_label < len(class_names):
            raise ValueError(f"split manifest class_label outside dense taxonomy: {dense_label}")
        if class_names[dense_label] != name or int(labels[index]) != dense_label:
            raise ValueError("split manifest label/name does not align with cached dense labels")
    leak_fields = ("source_group", "component_id", "source_fingerprint_v1", "source_image_sha256", "decoded_pixel_sha256", "source_pixel_sha256", "source_image_pixel_sha256")
    for field in leak_fields:
        train_values = {str(record[field]) for record in train_records if record.get(field) not in (None, "")}
        eval_values = {str(record[field]) for record in eval_records if record.get(field) not in (None, "")}
        overlap = sorted(train_values & eval_values)
        if overlap:
            raise ValueError(f"train/evaluation source leakage through {field}: {overlap[:3]}")
    return [_extract_class_name(record) for record in eval_records]


def _tensor_to_batches(tensor: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [tensor[start:start + batch_size] for start in range(0, tensor.size(0), batch_size)]


@torch.no_grad()
def _project_features(package: ModelPackage, features: torch.Tensor, *, batch_size: int, device: torch.device) -> torch.Tensor:
    model = ProjectionHead(input_dim=package.hidden_dim, embedding_dim=package.embedding_dim).to(device)
    model.load_state_dict(torch.load(package.projection_path, map_location="cpu", weights_only=True))
    model.eval()
    return torch.cat([model(batch.to(device)).cpu() for batch in _tensor_to_batches(features, batch_size)], dim=0) if features.numel() else torch.empty((0, package.embedding_dim), dtype=features.dtype)


def _compute_refit_prototypes(embeddings: torch.Tensor, labels: torch.Tensor, *, class_count: int) -> torch.Tensor:
    rows, missing = [], []
    for label in range(class_count):
        mask = labels.eq(label)
        if not mask.any():
            missing.append(label)
        else:
            rows.append(F.normalize(embeddings[mask].mean(dim=0), p=2, dim=-1))
    if missing:
        raise ValueError(f"train split is missing class(es) required for refit prototypes: {missing[:10]}")
    return torch.stack(rows)


def _evaluate_package(package: ModelPackage, features: torch.Tensor, labels: torch.Tensor, train_indices: list[int], eval_indices: list[int], truth_names: list[str], *, prototype_mode: str, batch_size: int, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eval_embeddings = _project_features(package, features[eval_indices], batch_size=batch_size, device=device)
    if prototype_mode == STORED_PROTOTYPES:
        prototypes, prototype_names, prototype_labels = package.prototypes, package.prototype_names, package.class_labels
        source = {"mode": STORED_PROTOTYPES, "path": str(package.prototypes_path), "sha256": sha256_file(package.prototypes_path)}
    else:
        train_embeddings = _project_features(package, features[train_indices], batch_size=batch_size, device=device)
        prototypes = _compute_refit_prototypes(train_embeddings, labels[train_indices], class_count=package.num_classes)
        prototype_names, prototype_labels = package.class_names, list(range(package.num_classes))
        source = {"mode": REFIT_PROTOTYPES, "path": None, "sha256": None, "train_examples": len(train_indices)}
    scores = torch.mm(eval_embeddings, prototypes.t())
    top3_values, top3_indices = scores.topk(min(3, package.num_classes), dim=1)
    predicted_names = [prototype_names[index] for index in top3_indices[:, 0].tolist()]
    top3_names = [[prototype_names[index] for index in row] for row in top3_indices.tolist()]
    top1_correct = [predicted == truth for predicted, truth in zip(predicted_names, truth_names)]
    top3_correct = [truth in predicted for truth, predicted in zip(truth_names, top3_names)]
    coverage = [float(value) >= package.rejection_threshold for value in top3_values[:, 0].tolist()]
    per_class = []
    for name in sorted(set(truth_names)):
        positions = [index for index, truth in enumerate(truth_names) if truth == name]
        per_class.append((sum(top1_correct[index] for index in positions) / len(positions), sum(top3_correct[index] for index in positions) / len(positions), sum(coverage[index] for index in positions) / len(positions)))
    predictions = [{"class_name": predicted_names[index], "class_label": prototype_labels[top3_indices[index, 0].item()], "confidence": float(top3_values[index, 0]), "rejected": not coverage[index], "top_k": [{"class_name": name, "class_label": prototype_labels[label], "confidence": float(value)} for name, label, value in zip(top3_names[index], top3_indices[index].tolist(), top3_values[index].tolist())]} for index in range(len(truth_names))]
    metrics = {"package_root": str(package.root), "config_path": str(package.config_path), "weights_dir": str(package.weights_dir), "prototypes_path": str(package.prototypes_path), "projection_sha256": sha256_file(package.projection_path), "prototype_source": source, "num_classes": package.num_classes, "class_count": package.num_classes, "hidden_dim": package.hidden_dim, "embedding_dim": package.embedding_dim, "rejection_threshold": package.rejection_threshold, "train_examples": len(train_indices), "eval_examples": len(truth_names), "top1_micro": sum(top1_correct) / len(truth_names) if truth_names else 0.0, "top3_micro": sum(top3_correct) / len(truth_names) if truth_names else 0.0, "top1_macro": sum(row[0] for row in per_class) / len(per_class) if per_class else 0.0, "top3_macro": sum(row[1] for row in per_class) / len(per_class) if per_class else 0.0, "coverage": sum(coverage) / len(truth_names) if truth_names else 0.0, "coverage_macro": sum(row[2] for row in per_class) / len(per_class) if per_class else 0.0}
    return metrics, predictions


def _label_contract(package: ModelPackage) -> dict[str, int]:
    return dict(sorted(zip(package.prototype_names, package.class_labels)))


def benchmark_model_replacement(*, features_path: Path, split_manifest_path: Path, runtime_dir: Path, candidate_dir: Path, output_path: Path | None = None, batch_size: int = DEFAULT_BATCH_SIZE, device: str = "cpu", prototype_mode: str = STORED_PROTOTYPES, eval_split: str | None = None) -> dict[str, Any]:
    if prototype_mode not in {STORED_PROTOTYPES, REFIT_PROTOTYPES}:
        raise ValueError(f"prototype_mode must be {STORED_PROTOTYPES!r} or {REFIT_PROTOTYPES!r}")
    payload = torch.load(features_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("features"), torch.Tensor) or not isinstance(payload.get("labels"), torch.Tensor):
        raise ValueError("cached features payload must contain tensors named features and labels")
    if prototype_mode == STORED_PROTOTYPES and payload.get("preprocessing") != PREPROCESSING_VERSION:
        raise ValueError("feature cache preprocessing is not runtime-aligned; rebuild the cache")
    features, labels = payload["features"], payload["labels"].to(torch.long)
    if features.ndim != 2 or labels.ndim != 1 or features.size(0) != labels.size(0):
        raise ValueError("cached features and labels must be aligned 2D/1D tensors")
    image_paths = [str(value) for value in payload["image_paths"]] if isinstance(payload.get("image_paths"), list) else None
    if image_paths is not None and len(image_paths) != features.size(0):
        raise ValueError("cached image_paths must align with features")
    records = _load_split_records(split_manifest_path)
    if not records:
        raise ValueError("split manifest contains no rows")
    runtime_package, candidate_package = _load_model_package(runtime_dir), _load_model_package(candidate_dir)
    if runtime_package.class_names != candidate_package.class_names:
        raise ValueError("runtime and candidate config class orders differ")
    eval_split = eval_split or _select_eval_split(records)
    if eval_split in TRAIN_SPLIT_NAMES:
        raise ValueError("evaluation split must not be train")
    train_records = [record for record in records if _extract_split_name(record) in TRAIN_SPLIT_NAMES]
    eval_records = [record for record in records if _extract_split_name(record) == eval_split]
    if not train_records or not eval_records:
        raise ValueError("split manifest requires both train and evaluation rows")
    train_indices = _resolve_indices(train_records, features_len=features.size(0), image_paths=image_paths, row_ids=payload.get("row_ids"))
    eval_indices = _resolve_indices(eval_records, features_len=features.size(0), image_paths=image_paths, row_ids=payload.get("row_ids"))
    truth_names = _validate_split_integrity(train_records, eval_records, train_indices, eval_indices, labels, runtime_package.class_names)
    device_obj = torch.device(device)
    runtime_metrics, runtime_predictions = _evaluate_package(runtime_package, features, labels, train_indices, eval_indices, truth_names, prototype_mode=prototype_mode, batch_size=batch_size, device=device_obj)
    candidate_metrics, candidate_predictions = _evaluate_package(candidate_package, features, labels, train_indices, eval_indices, truth_names, prototype_mode=prototype_mode, batch_size=batch_size, device=device_obj)
    split_counts: dict[str, int] = {}
    for record in records:
        split = _extract_split_name(record)
        split_counts[split] = split_counts.get(split, 0) + 1
    runtime_contract, candidate_contract = _label_contract(runtime_package), _label_contract(candidate_package)
    paired = [{"feature_index": index, "truth_class_name": truth_names[position], "truth_dense_label": _extract_label(eval_records[position]), "runtime": runtime_predictions[position], "candidate": candidate_predictions[position]} for position, index in enumerate(eval_indices)]
    report = {"schema_version": BENCHMARK_SCHEMA_VERSION, "prototype_mode": prototype_mode, "inputs": {"features_path": str(features_path.resolve()), "features_sha256": sha256_file(features_path), "split_manifest_path": str(split_manifest_path.resolve()), "split_manifest_sha256": sha256_file(split_manifest_path), "runtime_dir": str(runtime_package.root), "candidate_dir": str(candidate_package.root)}, "class_count": runtime_package.num_classes, "eval_split": eval_split, "split_counts": split_counts, "train_examples": len(train_indices), "eval_examples": len(eval_indices), "numeric_label_contract": {"comparison_basis": "class_name", "equal": runtime_contract == candidate_contract, "runtime_sha256": _hash_json(runtime_contract), "candidate_sha256": _hash_json(candidate_contract), "runtime": runtime_contract, "candidate": candidate_contract}, "models": {"runtime": runtime_metrics, "candidate": candidate_metrics}, "comparison": {"top1_micro_delta": candidate_metrics["top1_micro"] - runtime_metrics["top1_micro"], "top1_macro_delta": candidate_metrics["top1_macro"] - runtime_metrics["top1_macro"], "top3_micro_delta": candidate_metrics["top3_micro"] - runtime_metrics["top3_micro"], "top3_macro_delta": candidate_metrics["top3_macro"] - runtime_metrics["top3_macro"], "coverage_delta": candidate_metrics["coverage"] - runtime_metrics["coverage"], "coverage_macro_delta": candidate_metrics["coverage_macro"] - runtime_metrics["coverage_macro"]}, "paired_predictions": paired}
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path, help="Cached features.pt tensor dump.")
    parser.add_argument("--split-manifest", required=True, type=Path, help="JSON manifest with train/eval rows.")
    parser.add_argument("--runtime-dir", required=True, type=Path, help="Runtime classifier package or its root.")
    parser.add_argument("--candidate-dir", required=True, type=Path, help="Candidate classifier package or its root.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--prototype-mode", choices=[STORED_PROTOTYPES, REFIT_PROTOTYPES], default=STORED_PROTOTYPES, help="Use stored package prototypes (default) or the diagnostic train-split refit.")
    parser.add_argument("--eval-split", choices=["dev", "locked_test", "test", "val"], help="Choose the holdout explicitly; otherwise dev takes priority.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = benchmark_model_replacement(features_path=args.features, split_manifest_path=args.split_manifest, runtime_dir=args.runtime_dir, candidate_dir=args.candidate_dir, output_path=args.output, batch_size=args.batch_size, device=args.device, prototype_mode=args.prototype_mode, eval_split=args.eval_split)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
