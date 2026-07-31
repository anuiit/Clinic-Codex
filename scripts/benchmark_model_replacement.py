"""Benchmark runtime vs candidate classifier packages on cached features.

The benchmark is intentionally leakage-resistant:
1. It loads a cached ``features.pt`` tensor dump.
2. It reads a split manifest JSON to recover train/eval membership.
3. It reconstructs class prototypes from the train split only.
4. It compares runtime and candidate packages on the evaluation split.

The output is a single JSON report with top-1/top-3 micro/macro metrics and
coverage for each package.
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

BENCHMARK_SCHEMA_VERSION = "benchmark-model-replacement.v1"
TRAIN_SPLIT_NAMES = {"train"}
EVAL_SPLIT_PRIORITY = ("test", "val", "evaluation", "eval")
DEFAULT_BATCH_SIZE = 4096


@dataclass(frozen=True)
class ModelPackage:
    root: Path
    config_path: Path
    weights_dir: Path
    projection_path: Path
    prototypes_path: Path
    class_names: list[str]
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


def _load_json_value(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_package_root(path: Path) -> Path:
    path = path.resolve()
    direct_config = path / "config.json"
    direct_weights = path / "weights"
    if direct_config.is_file() and (direct_weights / "projection.pt").is_file() and (direct_weights / "prototypes.pt").is_file():
        return path

    runtime_dir = path / "runtime"
    runtime_config = runtime_dir / "config.json"
    runtime_weights = runtime_dir / "weights"
    if runtime_config.is_file() and (runtime_weights / "projection.pt").is_file() and (runtime_weights / "prototypes.pt").is_file():
        return runtime_dir

    raise FileNotFoundError(
        "package directory must contain config.json + weights/ or runtime/config.json + runtime/weights/ "
        f"(got {path})"
    )


def _load_model_package(path: Path) -> ModelPackage:
    root = _resolve_package_root(path)
    config_path = root / "config.json"
    weights_dir = root / "weights"
    projection_path = weights_dir / "projection.pt"
    prototypes_path = weights_dir / "prototypes.pt"
    config = _load_json_value(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"expected JSON object: {config_path}")
    class_names = load_runtime_class_order(config_path)
    num_classes = int(config.get("num_classes", len(class_names)))
    if len(class_names) != num_classes:
        raise ValueError(f"runtime config class_names length does not match num_classes: {config_path}")
    hidden_dim = int(config["hidden_dim"])
    embedding_dim = int(config["embedding_dim"])
    rejection_threshold = float(config.get("rejection_threshold", 0.35))
    return ModelPackage(
        root=root,
        config_path=config_path,
        weights_dir=weights_dir,
        projection_path=projection_path,
        prototypes_path=prototypes_path,
        class_names=class_names,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        rejection_threshold=rejection_threshold,
    )


def _normalise_split_name(value: Any) -> str:
    return str(value).strip().lower()


def _get_record_field(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    raise KeyError(f"split manifest row is missing required field(s): {', '.join(names)}")


def _load_split_records(path: Path) -> list[dict[str, Any]]:
    payload = _load_json_value(path)
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        records = payload["rows"]
    elif isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        records = payload["samples"]
    elif isinstance(payload, dict) and isinstance(payload.get("entries"), list):
        records = payload["entries"]
    elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
        records = payload["items"]
    elif isinstance(payload, dict) and isinstance(payload.get("split_rows"), list):
        records = payload["split_rows"]
    else:
        raise ValueError(f"unsupported split manifest schema: {path}")

    normalized: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"split manifest rows must be JSON objects: {path}")
        normalized.append(record)
    return normalized


def _extract_row_index(record: dict[str, Any]) -> int | None:
    for key in ("index", "row_index", "feature_index"):
        if key in record:
            return int(record[key])
    return None


def _extract_split_name(record: dict[str, Any]) -> str:
    return _normalise_split_name(_get_record_field(record, "dataset_split", "split", "subset", "partition"))


def _extract_label(record: dict[str, Any]) -> int:
    label = _get_record_field(record, "class_label", "label", "element_label")
    return int(label)


def _extract_path_key(record: dict[str, Any]) -> str | None:
    for key in ("image_path", "output_path", "source_path", "path"):
        if key in record and record[key] is not None:
            return str(record[key])
    return None


def _resolve_indices(
    records: list[dict[str, Any]],
    *,
    features_len: int,
    image_paths: list[str] | None,
) -> list[int]:
    if all(_extract_row_index(record) is not None for record in records):
        indices = [int(_extract_row_index(record)) for record in records]
    elif image_paths is not None:
        path_to_index = {str(path): idx for idx, path in enumerate(image_paths)}
        indices = []
        for record in records:
            path_key = _extract_path_key(record)
            if path_key is None:
                raise ValueError("split manifest rows need an index or a path field to match cached features")
            try:
                indices.append(path_to_index[path_key])
            except KeyError as exc:
                raise ValueError(f"split manifest path not found in features payload: {path_key}") from exc
    else:
        if len(records) > features_len:
            raise ValueError("split manifest has more rows than cached features")
        indices = list(range(len(records)))

    if len(set(indices)) != len(indices):
        raise ValueError("split manifest maps multiple rows to the same cached feature index")
    for index in indices:
        if index < 0 or index >= features_len:
            raise ValueError(f"cached feature index out of bounds: {index}")
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


def _tensor_to_batches(tensor: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    return [tensor[start:start + batch_size] for start in range(0, tensor.size(0), batch_size)]


@torch.no_grad()
def _project_features(package: ModelPackage, features: torch.Tensor, *, batch_size: int, device: torch.device) -> torch.Tensor:
    model = ProjectionHead(input_dim=package.hidden_dim, embedding_dim=package.embedding_dim).to(device)
    state_dict = torch.load(package.projection_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    outputs: list[torch.Tensor] = []
    for batch in _tensor_to_batches(features, batch_size):
        outputs.append(model(batch.to(device)).cpu())
    return torch.cat(outputs, dim=0) if outputs else torch.empty((0, package.embedding_dim), dtype=features.dtype)


def _compute_prototypes(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_count: int,
) -> torch.Tensor:
    prototypes = []
    missing = []
    for class_label in range(class_count):
        mask = labels == class_label
        if not mask.any():
            missing.append(class_label)
            continue
        centroid = embeddings[mask].mean(dim=0)
        prototypes.append(F.normalize(centroid, p=2, dim=-1))
    if missing:
        raise ValueError(f"train split is missing class(es) required for 286-way prototypes: {missing[:10]}")
    return torch.stack(prototypes, dim=0)


def _evaluate_package(
    package: ModelPackage,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_indices: list[int],
    eval_indices: list[int],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    eval_features = features[eval_indices]
    eval_labels = labels[eval_indices]

    train_embeddings = _project_features(package, train_features, batch_size=batch_size, device=device)
    eval_embeddings = _project_features(package, eval_features, batch_size=batch_size, device=device)
    prototypes = _compute_prototypes(train_embeddings, train_labels, class_count=package.num_classes)

    similarities = torch.mm(eval_embeddings, prototypes.t())
    top3_k = min(3, package.num_classes)
    top3_values, top3_indices = similarities.topk(top3_k, dim=1)
    top1_indices = top3_indices[:, 0]
    top1_values = top3_values[:, 0]

    top1_correct = top1_indices.eq(eval_labels)
    top3_correct = top3_indices.eq(eval_labels.unsqueeze(1)).any(dim=1)
    coverage = top1_values.ge(package.rejection_threshold)

    per_class_top1 = []
    per_class_top3 = []
    per_class_coverage = []
    for class_label in range(package.num_classes):
        mask = eval_labels == class_label
        if not mask.any():
            continue
        per_class_top1.append(top1_correct[mask].float().mean().item())
        per_class_top3.append(top3_correct[mask].float().mean().item())
        per_class_coverage.append(coverage[mask].float().mean().item())

    return {
        "package_root": str(package.root),
        "config_path": str(package.config_path),
        "weights_dir": str(package.weights_dir),
        "prototypes_path": str(package.prototypes_path),
        "num_classes": package.num_classes,
        "class_count": package.num_classes,
        "hidden_dim": package.hidden_dim,
        "embedding_dim": package.embedding_dim,
        "rejection_threshold": package.rejection_threshold,
        "train_examples": int(train_labels.numel()),
        "eval_examples": int(eval_labels.numel()),
        "top1_micro": float(top1_correct.float().mean().item()) if eval_labels.numel() else 0.0,
        "top3_micro": float(top3_correct.float().mean().item()) if eval_labels.numel() else 0.0,
        "top1_macro": float(sum(per_class_top1) / len(per_class_top1)) if per_class_top1 else 0.0,
        "top3_macro": float(sum(per_class_top3) / len(per_class_top3)) if per_class_top3 else 0.0,
        "coverage": float(coverage.float().mean().item()) if eval_labels.numel() else 0.0,
        "coverage_macro": float(sum(per_class_coverage) / len(per_class_coverage)) if per_class_coverage else 0.0,
    }


def benchmark_model_replacement(
    *,
    features_path: Path,
    split_manifest_path: Path,
    runtime_dir: Path,
    candidate_dir: Path,
    output_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "cpu",
) -> dict[str, Any]:
    features_payload = torch.load(features_path, map_location="cpu", weights_only=True)
    features = features_payload["features"]
    labels = features_payload["labels"].to(torch.long)
    if features.ndim != 2:
        raise ValueError("cached features tensor must be 2D")
    if labels.ndim != 1:
        raise ValueError("cached labels tensor must be 1D")
    if features.size(0) != labels.size(0):
        raise ValueError("cached features and labels must have the same row count")

    image_paths = None
    if isinstance(features_payload.get("image_paths"), list):
        image_paths = [str(item) for item in features_payload["image_paths"]]

    records = _load_split_records(split_manifest_path)
    if not records:
        raise ValueError("split manifest contains no rows")

    runtime_package = _load_model_package(runtime_dir)
    candidate_package = _load_model_package(candidate_dir)
    if runtime_package.class_names != candidate_package.class_names:
        raise ValueError("runtime and candidate class orders differ")

    eval_split = _select_eval_split(records)
    train_records = [record for record in records if _extract_split_name(record) in TRAIN_SPLIT_NAMES]
    eval_records = [record for record in records if _extract_split_name(record) == eval_split]
    if not train_records:
        raise ValueError("split manifest contains no train rows")
    if not eval_records:
        raise ValueError(f"split manifest contains no rows for evaluation split: {eval_split}")

    train_indices = _resolve_indices(train_records, features_len=features.size(0), image_paths=image_paths)
    eval_indices = _resolve_indices(eval_records, features_len=features.size(0), image_paths=image_paths)

    split_counts: dict[str, int] = {}
    for record in records:
        split = _extract_split_name(record)
        split_counts[split] = split_counts.get(split, 0) + 1

    device_obj = torch.device(device)
    runtime_metrics = _evaluate_package(
        runtime_package,
        features,
        labels,
        train_indices,
        eval_indices,
        batch_size=batch_size,
        device=device_obj,
    )
    candidate_metrics = _evaluate_package(
        candidate_package,
        features,
        labels,
        train_indices,
        eval_indices,
        batch_size=batch_size,
        device=device_obj,
    )

    report = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "inputs": {
            "features_path": str(features_path.resolve()),
            "features_sha256": sha256_file(features_path),
            "split_manifest_path": str(split_manifest_path.resolve()),
            "split_manifest_sha256": sha256_file(split_manifest_path),
            "runtime_dir": str(runtime_package.root),
            "candidate_dir": str(candidate_package.root),
        },
        "class_count": runtime_package.num_classes,
        "eval_split": eval_split,
        "split_counts": split_counts,
        "train_examples": len(train_indices),
        "eval_examples": len(eval_indices),
        "models": {
            "runtime": runtime_metrics,
            "candidate": candidate_metrics,
        },
        "comparison": {
            "top1_micro_delta": candidate_metrics["top1_micro"] - runtime_metrics["top1_micro"],
            "top1_macro_delta": candidate_metrics["top1_macro"] - runtime_metrics["top1_macro"],
            "top3_micro_delta": candidate_metrics["top3_micro"] - runtime_metrics["top3_micro"],
            "top3_macro_delta": candidate_metrics["top3_macro"] - runtime_metrics["top3_macro"],
            "coverage_delta": candidate_metrics["coverage"] - runtime_metrics["coverage"],
            "coverage_macro_delta": candidate_metrics["coverage_macro"] - runtime_metrics["coverage_macro"],
        },
    }

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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        report = benchmark_model_replacement(
            features_path=args.features,
            split_manifest_path=args.split_manifest,
            runtime_dir=args.runtime_dir,
            candidate_dir=args.candidate_dir,
            output_path=args.output,
            batch_size=args.batch_size,
            device=args.device,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
