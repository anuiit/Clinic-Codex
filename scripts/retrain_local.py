#!/usr/bin/env python3
"""Create a local candidate from the shipped prior and current approved crops.

No optimizer, private corpus, network access, or activation. Replaying all current
approvals against the fixed prior makes reruns idempotent and handles revocations.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from PIL import Image

from backend.codex_model.classifier import PREPROCESSING_VERSION, _ProjectionHead, _preprocess_image
from backend.codex_pipeline.scripts.export_model import _prototype_contract, export_model
from backend.services.annotation_review import AnnotationReviewStore
from backend.services.model_registry import ModelRegistry
from backend.services.training_jobs import _process_is_alive, _read_lock_pid
from scripts.build_training_snapshot import pixel_sha256
from scripts.pin_dinov2 import DINOV2_WEIGHTS_SHA256, load_backbone, sha256_file


def merge_prior(prior: dict, embeddings: torch.Tensor, labels: list[int]) -> dict:
    """Recover historical sums from count/cosine variance, then add unique crops."""
    ordered, _ = _prototype_contract(prior, label="shipped base")
    if embeddings.shape != (len(labels), prior["prototypes"].shape[1]):
        raise ValueError("new embeddings and labels do not align")
    if not torch.isfinite(embeddings).all() or not torch.allclose(
        embeddings.norm(dim=1), torch.ones(len(labels)), atol=1e-4
    ):
        raise ValueError("new embeddings must be finite unit vectors")
    if not set(labels).issubset(ordered):
        raise ValueError("annotations contain classes absent from the base")
    result = copy.deepcopy(prior)
    for index, label in enumerate(ordered):
        meta = prior.get("class_meta", {}).get(label, {})
        count, variance = meta.get("count"), meta.get("variance")
        if variance is None and count == 1:
            variance = 0.0
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"base class {label}: missing valid historical count")
        if not isinstance(variance, (int, float)) or not math.isfinite(variance) or not 0 <= variance <= 1:
            raise ValueError(f"base class {label}: missing valid cosine variance")
        selected = [i for i, value in enumerate(labels) if value == label]
        if not selected:
            continue
        vector = count * (1 - variance) * prior["prototypes"][index]
        vector = vector + embeddings[selected].sum(dim=0)
        norm = vector.norm().item()
        if not math.isfinite(norm) or norm <= 1e-8:
            raise ValueError(f"base class {label}: degenerate updated prototype")
        result["prototypes"][index] = vector / norm
        result["class_meta"][label] = {
            **meta, "count": count + len(selected),
            "variance": max(0.0, 1 - norm / (count + len(selected))),
        }
    _prototype_contract(result, label="local candidate")
    return result


def capture_approvals(store: AnnotationReviewStore, destination: Path, names: dict[int, str]) -> dict:
    """Capture bytes first; reject concurrent edits and contradictory duplicate pixels."""
    destination.mkdir(parents=True, exist_ok=False)
    review_hash = sha256_file(store.manifest_path)
    records = list(store.iter_approved_annotations())
    if not records:
        raise ValueError("no current approved annotations; approve an element first")
    labels_by_name = {name: label for label, name in names.items()}
    rows, pixels_seen = [], {}
    for record in records:
        if record["class_name"] not in labels_by_name:
            raise ValueError(f"unknown base class: {record['class_name']}")
        source = Path(record["crop_path"])
        source_hash = sha256_file(source)
        source_image = Path(record["image_path"])
        image_hash = sha256_file(source_image)
        target = destination / f"{len(rows):06d}.image"
        shutil.copyfile(source, target)
        if sha256_file(target) != source_hash or sha256_file(source) != source_hash:
            raise ValueError("annotation crop changed during capture; retry")
        pixel_hash = pixel_sha256(target)
        label = labels_by_name[record["class_name"]]
        if pixel_hash in pixels_seen:
            if pixels_seen[pixel_hash] != label:
                raise ValueError("identical crop pixels have conflicting approved classes")
            target.unlink()
            continue
        pixels_seen[pixel_hash] = label
        rows.append({
            **record, "class_label": label, "snapshot_crop": target.name,
            "crop_sha256": source_hash, "pixel_sha256": pixel_hash,
            "source_image_sha256": image_hash,
        })
    if sha256_file(store.manifest_path) != review_hash or list(store.iter_approved_annotations()) != records:
        raise ValueError("annotation reviews changed during capture; retry")
    for record in rows:
        if (sha256_file(Path(record["crop_path"])) != record["crop_sha256"]
                or sha256_file(Path(record["image_path"])) != record["source_image_sha256"]):
            raise ValueError("annotation source changed during capture; retry")
    shutil.copyfile(store.manifest_path, destination / "review-index.json")
    if sha256_file(destination / "review-index.json") != review_hash:
        raise ValueError("annotation review index changed during capture; retry")
    snapshot = {
        "schema_version": "local-approved-snapshot.v1", "review_index_sha256": review_hash,
        "approved_count": len(records), "unique_count": len(rows),
        "duplicate_count": len(records) - len(rows), "rows": rows,
        "evaluation_scope": "training_fit_only_no_holdout",
    }
    (destination / "manifest.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    return snapshot


def create_candidate(args) -> dict:
    registry = ModelRegistry(args.registry_dir, repo_root=ROOT, runtime_model_dir=args.base_model_dir)
    version = registry.version_dir(args.version_id)
    version.mkdir(parents=True, exist_ok=False)
    inputs = version / "inputs"
    inputs.mkdir()
    protected = [args.prior, args.base_model_dir / "config.json",
                 args.base_model_dir / "weights/projection.pt", args.base_model_dir / "weights/prototypes.pt",
                 args.backbone_manifest]
    before = {str(path): sha256_file(path) for path in protected}
    for path, name in zip(protected, ("prior.pt", "config.json", "projection.pt", "active-prototypes.pt", "backbone.json")):
        shutil.copyfile(path, inputs / name)
        if sha256_file(inputs / name) != before[str(path)]:
            raise ValueError("model changed during capture; retry")
    prior = torch.load(inputs / "prior.pt", map_location="cpu", weights_only=True)
    ordered, names = _prototype_contract(prior, label="shipped base")
    active = torch.load(inputs / "active-prototypes.pt", map_location="cpu", weights_only=True)
    active_labels, active_names = _prototype_contract(active, label="active base")
    if active_labels != ordered or active_names != names:
        raise ValueError("installed taxonomy differs from the shipped base")
    projection_state = torch.load(inputs / "projection.pt", map_location="cpu", weights_only=True)
    if projection_state.keys() != prior["model_state_dict"].keys() or any(
        not torch.equal(projection_state[key], prior["model_state_dict"][key]) for key in projection_state
    ):
        raise ValueError("installed projection differs from the shipped prior; use the advanced snapshot workflow")
    config = json.loads((inputs / "config.json").read_text(encoding="utf-8"))
    if (config["backbone"] != "dinov2_vits14" or config["image_size"] != 224
            or config["hidden_dim"] != prior["hidden_dim"] or config["embedding_dim"] != prior["embedding_dim"]
            or config["class_names"] != [names[label] for label in ordered]):
        raise ValueError("installed configuration is incompatible with the shipped prior")
    pin = json.loads((inputs / "backbone.json").read_text(encoding="utf-8"))
    if pin.get("weights_sha256") != DINOV2_WEIGHTS_SHA256:
        raise ValueError("local retraining requires the official shipped-base DINOv2 weights")
    # Validate every prior class, including classes untouched by this run.
    merge_prior(prior, torch.empty((0, prior["embedding_dim"])), [])
    snapshot = capture_approvals(
        AnnotationReviewStore(args.annotations_dir, args.review_manifest), version / "annotations", names,
    )
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(args.device)
    backbone, backbone_provenance = load_backbone(config["backbone"], device, inputs / "backbone.json")
    if args.dry_run:
        print(f"Dry-run passed: {snapshot['unique_count']} unique approvals, base and backbone verified.")
        return {"dry_run": True, "unique_count": snapshot["unique_count"]}
    projection = _ProjectionHead(prior["hidden_dim"], prior["embedding_dim"]).to(device).eval()
    projection.load_state_dict(projection_state, strict=True)
    features = []
    with torch.inference_mode():
        for offset in range(0, len(snapshot["rows"]), args.batch_size):
            batch = []
            for row in snapshot["rows"][offset:offset + args.batch_size]:
                path = version / "annotations" / row["snapshot_crop"]
                if sha256_file(path) != row["crop_sha256"]:
                    raise ValueError("captured crop checksum mismatch")
                with Image.open(path) as image:
                    batch.append(_preprocess_image(image, config["image_size"]).squeeze(0))
            features.append(projection(backbone(torch.stack(batch).to(device))).cpu())
    embeddings = torch.cat(features)
    labels = [row["class_label"] for row in snapshot["rows"]]
    updated = merge_prior(prior, embeddings, labels)
    source = version / "prototypes.pt"
    torch.save(updated, source)
    runtime = version / "runtime"
    export_model(source, runtime / "weights", inputs / "config.json",
                 config_out_path=runtime / "config.json", base_model_dir=args.base_model_dir)
    # Preserve the exact frozen projection bytes, not just equivalent tensors.
    shutil.copyfile(inputs / "projection.pt", runtime / "weights/projection.pt")
    output = torch.load(runtime / "weights/prototypes.pt", map_location="cpu", weights_only=True)
    if not torch.equal(output["prototypes"], updated["prototypes"]):
        raise ValueError("exported candidate differs from computed prototypes")
    expected = torch.tensor(labels)
    def score(prototypes):
        predictions = torch.tensor(ordered)[(embeddings @ prototypes.T).argmax(dim=1)]
        return int((predictions == expected).sum())
    report = {
        "schema_version": "local-retraining-result.v1", "training_mode": "local_prior",
        "evaluation_scope": "training_fit_only_no_holdout", "generalization_validated": False,
        "approved_count": snapshot["approved_count"], "unique_count": len(labels),
        "duplicate_count": snapshot["duplicate_count"],
        "updated_classes": sorted({names[label] for label in labels}),
        "base_correct": score(prior["prototypes"]), "active_correct": score(active["prototypes"]),
        "candidate_correct": score(output["prototypes"]), "device": str(device),
        "preprocessing": PREPROCESSING_VERSION, "inputs_sha256": before,
        "backbone": backbone_provenance, "model_version_id": args.version_id,
        "activation": "none",
    }
    evaluation = version / "evaluation"
    evaluation.mkdir()
    (evaluation / "local.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if any(sha256_file(Path(path)) != digest for path, digest in before.items()):
        raise ValueError("base or backbone configuration changed during training; candidate not registered")
    registry.write_manifest(
        args.version_id, artifact_paths=[p for p in version.rglob("*") if p.is_file()],
        metadata={
            "training": {"mode": "local_prior", "projection_frozen": True, "preprocessing": PREPROCESSING_VERSION},
            "data": {"approved_count": len(labels), "holdout_count": 0},
            "metrics": report,
            "promotion": {"blocked": True, "reason": "Local annotations only: no independent holdout evaluation."},
        },
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


@contextmanager
def training_lock(backend_root: Path):
    # The OS releases this guard even after SIGKILL; retain the legacy PID file
    # so the advanced scripts and backend can still identify the active job.
    lock = backend_root / ".retrain.lock"
    with (backend_root / ".retrain.guard").open("a+b") as guard:
        if os.name == "nt":
            import msvcrt
            guard.seek(0)
            msvcrt.locking(guard.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if lock.exists():
            pid = _read_lock_pid(lock)
            if pid is not None and _process_is_alive(pid) is not False:
                raise RuntimeError(f"retraining already running (PID {pid})")
            lock.unlink()
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        try:
            yield
        finally:
            if _read_lock_pid(lock) == os.getpid():
                lock.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-root", type=Path, default=ROOT / "backend")
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--backbone-manifest", type=Path, required=True)
    parser.add_argument("--registry-dir", type=Path, required=True)
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 256:
        parser.error("batch size must be between 1 and 256")
    args.prior = args.backend_root / "prototypes/prototypes.pt"
    args.base_model_dir = args.backend_root / "codex_model"
    with training_lock(args.backend_root):
        create_candidate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
