#!/usr/bin/env python3
"""Ten-iteration Elements autoresearch loop on a source-disjoint external holdout.

The strict holdout is never used for training/prototypes. It is split into a
development validation subset and a sealed test subset. Model selection only
uses development validation; sealed test is evaluated once for the leader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from codex_pipeline.determinism import configure_determinism
from codex_pipeline.models.projection_head import ProjectionHead

DEFAULT_EXTERNAL_CACHE = BACKEND_ROOT / "model_registry/versions/20260711T220000Z-external286-weak-v3/training_data/precomputed/features.pt"
DEFAULT_LEGACY_CACHE = Path("/tmp/baseline-replacement-e2e/perclass-split/features_train_augmented_safe.pt")
DEFAULT_STRICT_RESULTS = REPO_ROOT / "reports/model-comparison-strict-unseen-20260729/element-results.json"
DEFAULT_HISTORICAL = BACKEND_ROOT / "codex_model/weights"
DEFAULT_CANDIDATE = BACKEND_ROOT / "model_registry/versions/20260729T020000Z-elements-refit-m5-teacher-safe/runtime/weights"
DEFAULT_RUN_DIR = REPO_ROOT / ".omc/autoresearch/elements-baseline-replacement/runs/20260729-external-strict-v4"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def resolved(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def class_names(cache: dict[str, Any]) -> dict[int, str]:
    names = cache.get("class_names")
    if not isinstance(names, dict):
        raise ValueError("feature cache must expose class_names")
    return {int(key): str(value) for key, value in names.items()}


def load_cache(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not {"features", "labels", "image_paths", "class_names"} <= payload.keys():
        raise ValueError(f"incomplete feature cache: {path}")
    if len(payload["features"]) != len(payload["labels"]) or len(payload["labels"]) != len(payload["image_paths"]):
        raise ValueError(f"misaligned feature cache: {path}")
    return payload


@dataclass
class Runtime:
    weights_dir: Path
    model: ProjectionHead
    prototypes: torch.Tenso
    labels: list[int]
    names: dict[int, str]


def load_runtime(weights_dir: Path, device: torch.device) -> Runtime:
    model = ProjectionHead(384, 128).to(device)
    model.load_state_dict(torch.load(weights_dir / "projection.pt", map_location=device, weights_only=False))
    model.eval()
    payload = torch.load(weights_dir / "prototypes.pt", map_location="cpu", weights_only=False)
    names = {int(key): str(value) for key, value in payload["class_names"].items()}
    labels = sorted(names)
    prototypes = payload["prototypes"].float().to(device)
    if prototypes.shape != (286, 128) or len(labels) != 286:
        raise ValueError(f"invalid runtime taxonomy/prototypes: {weights_dir}")
    return Runtime(weights_dir, model, F.normalize(prototypes, dim=1), labels, names)


def cache_positions(cache: dict[str, Any], runtime: Runtime) -> torch.Tensor:
    dense_names = class_names(cache)
    name_to_position = {runtime.names[label]: position for position, label in enumerate(runtime.labels)}
    missing = sorted(set(dense_names.values()) - set(name_to_position))
    if missing:
        raise ValueError(f"cache/runtime class mismatch: {missing[:5]}")
    mapping = torch.tensor([name_to_position[dense_names[index]] for index in range(len(dense_names))])
    return mapping[cache["labels"].long()]


def strict_split(external: dict[str, Any], strict_results_path: Path) -> dict[str, Any]:
    strict_rows = json.loads(strict_results_path.read_text(encoding="utf-8"))
    if len(strict_rows) != 569:
        raise ValueError(f"expected audited strict holdout of 569 rows, got {len(strict_rows)}")
    external_by_path = {resolved(path): index for index, path in enumerate(external["image_paths"])}
    strict_indices: list[int] = []
    row_by_index: dict[int, dict[str, Any]] = {}
    for row in strict_rows:
        path = resolved(row["path"])
        if path not in external_by_path:
            raise ValueError(f"strict holdout row missing from external cache: {path}")
        index = external_by_path[path]
        strict_indices.append(index)
        row_by_index[index] = row
    if len(set(strict_indices)) != 569:
        raise ValueError("strict holdout contains duplicate cache rows")

    by_class_group: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index in strict_indices:
        row = row_by_index[index]
        label = int(external["labels"][index])
        group = str(row.get("source_path") or row["path"]).casefold()
        by_class_group[label][group].append(index)

    dev: list[int] = []
    test: list[int] = []
    dev_groups: set[str] = set()
    test_groups: set[str] = set()
    for label, grouped in sorted(by_class_group.items()):
        groups = sorted(grouped, key=lambda value: stable_hash(f"{label}:{value}"))
        for position, group in enumerate(groups):
            if len(groups) == 1:
                use_dev = stable_hash(f"singleton:{label}:{group}") % 2 == 0
            else:
                use_dev = position % 2 == 0
            target = dev if use_dev else test
            target.extend(grouped[group])
            (dev_groups if use_dev else test_groups).add(group)

    if dev_groups & test_groups or set(dev) & set(test):
        raise ValueError("source-disjoint strict split failed")
    strict_set = set(strict_indices)
    train_external = [index for index in range(len(external["labels"])) if index not in strict_set]
    return {
        "external_train": sorted(train_external),
        "dev": sorted(dev),
        "test": sorted(test),
        "strict": sorted(strict_indices),
        "dev_group_count": len(dev_groups),
        "test_group_count": len(test_groups),
    }


@torch.inference_mode()
def project(model: ProjectionHead, features: torch.Tensor, device: torch.device, batch_size: int = 2048) -> torch.Tensor:
    model.eval()
    return torch.cat([model(batch.float().to(device)).cpu() for batch in features.split(batch_size)])


def blended_prototypes(
    model: ProjectionHead,
    legacy_features: torch.Tensor,
    legacy_labels: torch.Tensor,
    external_features: torch.Tensor,
    external_labels: torch.Tensor,
    external_weight: float,
    device: torch.device,
    base_prototypes: torch.Tensor | None = None,
) -> torch.Tensor:
    legacy_embeddings = project(model, legacy_features, device) if base_prototypes is None else None
    external_embeddings = project(model, external_features, device)
    rows: list[torch.Tensor] = []
    for label in range(286):
        legacy_centroid = (
            F.normalize(legacy_embeddings[legacy_labels == label].mean(0), dim=0)
            if legacy_embeddings is not None
            else F.normalize(base_prototypes[label].cpu(), dim=0)
        )
        mask = external_labels == label
        if bool(mask.any()) and external_weight > 0:
            external_centroid = F.normalize(external_embeddings[mask].mean(0), dim=0)
            centroid = F.normalize((1 - external_weight) * legacy_centroid + external_weight * external_centroid, dim=0)
        else:
            centroid = legacy_centroid
        rows.append(centroid)
    return torch.stack(rows)


@torch.inference_mode()
def metrics(
    model: ProjectionHead,
    prototypes: torch.Tensor,
    features: torch.Tensor,
    truth: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, float], torch.Tensor]:
    embeddings = project(model, features, device)
    scores = embeddings @ prototypes.cpu().T
    top3 = scores.topk(3, dim=1).indices
    predictions = top3[:, 0]
    per_class = []
    for label in truth.unique(sorted=True):
        mask = truth == label
        per_class.append(float((predictions[mask] == truth[mask]).float().mean()))
    result = {
        "top1": float((predictions == truth).float().mean()),
        "macro_top1": sum(per_class) / len(per_class),
        "top3": float((top3 == truth[:, None]).any(1).float().mean()),
        "mean_confidence": float(scores.max(1).values.mean()),
        "count": int(len(truth)),
        "class_count": int(truth.unique().numel()),
    }
    return result, predictions


def train_iteration(
    spec: dict[str, Any],
    initial_state: dict[str, torch.Tensor],
    teacher: ProjectionHead,
    features: torch.Tensor,
    labels: torch.Tensor,
    external_mask: torch.Tensor,
    device: torch.device,
) -> ProjectionHead:
    configure_determinism(int(spec["seed"]))
    model = ProjectionHead(384, 128).to(device)
    model.load_state_dict(initial_state)
    model.net[2].p = float(spec["dropout"])
    if spec.get("freeze_first", False):
        for parameter in model.net[0].parameters():
            parameter.requires_grad = False
    epochs = int(spec["epochs"])
    if epochs == 0:
        return model

    with torch.inference_mode():
        initial_embeddings = project(model, features, device)
        proxy_rows = [F.normalize(initial_embeddings[labels == label].mean(0), dim=0) for label in range(286)]
    proxies = torch.nn.Parameter(torch.stack(proxy_rows).to(device))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad] + [proxies]
    optimizer = torch.optim.AdamW(parameters, lr=float(spec["lr"]), weight_decay=float(spec["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    generator = torch.Generator().manual_seed(int(spec["seed"]))
    class_counts = torch.bincount(labels, minlength=286).float()
    sample_weights = torch.ones(len(labels))
    if spec["class_balanced"]:
        sample_weights = (class_counts.mean() / class_counts[labels]).clamp(max=10)
    sample_weights[external_mask] *= float(spec["external_weight"])
    batch_size = int(spec["batch_size"])
    teacher.eval()

    for _ in range(epochs):
        permutation = torch.randperm(len(labels), generator=generator)
        model.train()
        for batch_indices in permutation.split(batch_size):
            batch = features[batch_indices].float().to(device)
            target = labels[batch_indices].long().to(device)
            weights = sample_weights[batch_indices].to(device)
            embeddings = model(batch)
            logits = embeddings @ F.normalize(proxies, dim=1).T / float(spec["temperature"])
            ce = F.cross_entropy(logits, target, reduction="none", label_smoothing=float(spec["label_smoothing"]))
            loss = (ce * weights).sum() / weights.sum()
            teacher_weight = float(spec["teacher_weight"])
            if teacher_weight:
                with torch.no_grad():
                    teacher_embeddings = teacher(batch)
                loss = loss + teacher_weight * (1 - F.cosine_similarity(embeddings, teacher_embeddings).mean())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
        scheduler.step()
    model.eval()
    return model


def exact_mcnemar(candidate_correct: torch.Tensor, historical_correct: torch.Tensor) -> dict[str, Any]:
    candidate_only = int((candidate_correct & ~historical_correct).sum())
    historical_only = int((historical_correct & ~candidate_correct).sum())
    discordant = candidate_only + historical_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(candidate_only, historical_only) + 1)) / (2 ** discordant)
        p_value = min(1.0, 2 * tail)
    return {"candidate_only_correct": candidate_only, "historical_only_correct": historical_only, "discordant": discordant, "exact_p_value": p_value}


def specs() -> list[dict[str, Any]]:
    common = {"batch_size": 512, "temperature": 0.08, "weight_decay": 1e-6, "dropout": 0.0, "label_smoothing": 0.02, "class_balanced": True}
    variants = [
        ("candidate-prototypes-050", "candidate", 0, 1e-4, 0.0, 2.0, 0.50, False, 5101),
        ("historic-ce20-anchor050", "historical", 20, 1e-4, 0.50, 2.0, 0.50, False, 5102),
        ("historic-ce40-anchor025", "historical", 40, 1e-4, 0.25, 2.0, 0.50, False, 5103),
        ("candidate-ce40-anchor025", "candidate", 40, 1e-4, 0.25, 2.0, 0.50, False, 5104),
        ("historic-ce40-external4", "historical", 40, 2e-4, 0.50, 4.0, 0.75, False, 5105),
        ("historic-ce60-low-lr", "historical", 60, 5e-5, 0.50, 3.0, 0.75, False, 5106),
        ("historic-ce40-unbalanced", "historical", 40, 1e-4, 0.25, 3.0, 0.50, False, 5107),
        ("candidate-ce60-light-anchor", "candidate", 60, 5e-5, 0.10, 3.0, 0.75, False, 5108),
        ("historic-ce40-smooth050", "historical", 40, 1e-4, 0.35, 3.0, 0.60, False, 5109),
        ("historic-ce40-freeze-first", "historical", 40, 2e-4, 0.25, 3.0, 0.60, True, 5110),
    ]
    output = []
    for iteration, variant in enumerate(variants, 1):
        name, initialization, epochs, lr, teacher_weight, external_weight, prototype_weight, freeze_first, seed = variant
        item = {**common, "iteration": iteration, "name": name, "initialization": initialization, "epochs": epochs, "lr": lr, "teacher_weight": teacher_weight, "external_weight": external_weight, "prototype_external_weight": prototype_weight, "freeze_first": freeze_first, "seed": seed}
        if name.endswith("unbalanced"):
            item["class_balanced"] = False
        if name.endswith("smooth050"):
            item["label_smoothing"] = 0.05
        output.append(item)
    for offset, weight in enumerate((0.05, 0.10, 0.20, 0.30, 0.50, 0.75), start=11):
        output.append({
            **common,
            "iteration": offset,
            "name": f"historical-runtime-proto-{weight:.2f}",
            "initialization": "historical",
            "epochs": 0,
            "lr": 0.0,
            "teacher_weight": 0.0,
            "external_weight": 1.0,
            "prototype_external_weight": weight,
            "prototype_base": "historical",
            "freeze_first": False,
            "seed": 5200 + offset,
        })
    for offset, weight in enumerate((0.05, 0.10, 0.20, 0.30), start=17):
        output.append({
            **common,
            "iteration": offset,
            "name": f"candidate-runtime-proto-{weight:.2f}",
            "initialization": "candidate",
            "epochs": 0,
            "lr": 0.0,
            "teacher_weight": 0.0,
            "external_weight": 1.0,
            "prototype_external_weight": weight,
            "prototype_base": "candidate",
            "freeze_first": False,
            "seed": 5200 + offset,
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-cache", type=Path, default=DEFAULT_EXTERNAL_CACHE)
    parser.add_argument("--legacy-cache", type=Path, default=DEFAULT_LEGACY_CACHE)
    parser.add_argument("--strict-results", type=Path, default=DEFAULT_STRICT_RESULTS)
    parser.add_argument("--historical-weights", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--candidate-weights", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-iterations", type=int, default=10)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested autoresearch run")
    device = torch.device(args.device)
    run_dir = args.run_dir
    started_at = utc_now()
    deadline = time.time() + 6 * 60 * 60
    state_path = run_dir / "state.json"
    dump_json(state_path, {"status": "running", "run_id": run_dir.name, "mission": "elements-baseline-replacement", "started_at": started_at, "updated_at": started_at, "iteration": 0, "max_runtime_seconds": 21600})

    external = load_cache(args.external_cache)
    legacy = load_cache(args.legacy_cache)
    historical = load_runtime(args.historical_weights, device)
    candidate = load_runtime(args.candidate_weights, device)
    if historical.labels != candidate.labels or historical.names != candidate.names:
        raise ValueError("runtime taxonomies differ")
    split = strict_split(external, args.strict_results)
    dump_json(run_dir / "split-audit.json", {**{key: value for key, value in split.items() if key not in {"external_train", "dev", "test", "strict"}}, "external_total": len(external["labels"]), "external_train_count": len(split["external_train"]), "strict_count": len(split["strict"]), "dev_count": len(split["dev"]), "test_count": len(split["test"]), "external_cache_sha256": sha256_file(args.external_cache), "legacy_cache_sha256": sha256_file(args.legacy_cache), "strict_results_sha256": sha256_file(args.strict_results), "source_overlap_count": 0})

    external_truth = cache_positions(external, historical)
    legacy_truth = cache_positions(legacy, historical)
    external_train_indices = torch.tensor(split["external_train"])
    dev_indices = torch.tensor(split["dev"])
    test_indices = torch.tensor(split["test"])
    train_features = torch.cat([legacy["features"], external["features"][external_train_indices]])
    train_labels = torch.cat([legacy_truth, external_truth[external_train_indices]])
    external_mask = torch.zeros(len(train_labels), dtype=torch.bool)
    external_mask[len(legacy_truth):] = True
    dev_features = external["features"][dev_indices]
    dev_truth = external_truth[dev_indices]

    historical_dev, historical_dev_predictions = metrics(historical.model, historical.prototypes, dev_features, dev_truth, device)
    candidate_dev, _ = metrics(candidate.model, candidate.prototypes, dev_features, dev_truth, device)
    evaluator = {"schema_version": "autoresearch-external-evaluator.v1", "mission": "elements-baseline-replacement", "primary_metric": "source_disjoint_external_dev_top1", "required_output": {"pass": "boolean", "score": "number"}, "historical_dev": historical_dev, "previous_candidate_dev": candidate_dev, "selection": "strict improvement over prior dev leader; sealed test unavailable until final leader"}
    dump_json(run_dir / "evaluator.json", evaluator)

    historical_state = {key: value.detach().cpu() for key, value in historical.model.state_dict().items()}
    candidate_state = {key: value.detach().cpu() for key, value in candidate.model.state_dict().items()}
    teacher = historical.model
    incumbent = historical_dev["top1"]
    leader: dict[str, Any] | None = None
    decision_lines = ["# External strict autoresearch decision log", "", f"Started: {started_at}", f"Historical dev top-1: {historical_dev['top1']:.6f}", f"Previous candidate dev top-1: {candidate_dev['top1']:.6f}", ""]

    for spec in specs()[: args.max_iterations]:
        if time.time() >= deadline:
            break
        iteration_started = time.perf_counter()
        initial_state = historical_state if spec["initialization"] == "historical" else candidate_state
        model = train_iteration(spec, initial_state, teacher, train_features, train_labels, external_mask, device)
        prototype_base_name = spec.get("prototype_base")
        base_prototypes = (
            historical.prototypes
            if prototype_base_name == "historical"
            else candidate.prototypes
            if prototype_base_name == "candidate"
            else None
        )
        prototypes = blended_prototypes(
            model,
            legacy["features"],
            legacy_truth,
            external["features"][external_train_indices],
            external_truth[external_train_indices],
            float(spec["prototype_external_weight"]),
            device,
            base_prototypes=base_prototypes,
        )
        dev_result, dev_predictions = metrics(model, prototypes, dev_features, dev_truth, device)
        passed = bool(dev_result["top1"] > incumbent)
        checkpoint_path = run_dir / "checkpoints" / f"iteration-{spec['iteration']:04d}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "prototypes": prototypes.cpu(), "spec": spec}, checkpoint_path)
        evaluation = {"schema_version": "autoresearch-external-evaluation.v1", "iteration": spec["iteration"], "name": spec["name"], "pass": passed, "score": dev_result["top1"], "dev": dev_result, "historical_dev": historical_dev, "previous_incumbent": incumbent, "runtime_compatible": True, "prototype_class_count": 286, "train_feature_count": len(train_labels), "external_train_count": len(external_train_indices), "strict_holdout_used_for_training": False, "sealed_test_evaluated": False, "elapsed_seconds": time.perf_counter() - iteration_started, "spec": spec, "checkpoint_path": str(checkpoint_path)}
        dump_json(run_dir / "evaluations" / f"iteration-{spec['iteration']:04d}.json", evaluation)
        dump_json(run_dir / "specs" / f"iteration-{spec['iteration']:04d}.json", spec)
        decision_lines.extend([f"## Iteration {spec['iteration']:04d} — {spec['name']}", "", f"- Dev top-1: {dev_result['top1']:.6f}", f"- Dev macro top-1: {dev_result['macro_top1']:.6f}", f"- Decision: {'KEEP' if passed else 'REJECT'}", f"- Sealed test read: no", ""])
        if passed:
            incumbent = dev_result["top1"]
            leader = {"spec": spec, "model": model, "prototypes": prototypes, "checkpoint_path": checkpoint_path, "dev": dev_result, "dev_predictions": dev_predictions}
        dump_json(state_path, {"status": "running", "run_id": run_dir.name, "mission": "elements-baseline-replacement", "started_at": started_at, "updated_at": utc_now(), "iteration": spec["iteration"], "leader": leader["spec"]["name"] if leader else None, "max_runtime_seconds": 21600})

    if leader is None:
        raise RuntimeError("no iteration beat the historical development score")
    test_features = external["features"][test_indices]
    test_truth = external_truth[test_indices]
    historical_test, historical_test_predictions = metrics(historical.model, historical.prototypes, test_features, test_truth, device)
    leader_test, leader_test_predictions = metrics(leader["model"], leader["prototypes"], test_features, test_truth, device)
    previous_candidate_test, _ = metrics(candidate.model, candidate.prototypes, test_features, test_truth, device)
    paired = exact_mcnemar(leader_test_predictions == test_truth, historical_test_predictions == test_truth)
    final_pass = bool(leader["dev"]["top1"] > historical_dev["top1"] and leader_test["top1"] > historical_test["top1"])
    final = {"schema_version": "autoresearch-external-final.v1", "pass": final_pass, "score": leader_test["top1"] - historical_test["top1"], "selected": leader["spec"], "development": {"historical": historical_dev, "previous_candidate": candidate_dev, "selected": leader["dev"]}, "sealed_test": {"historical": historical_test, "previous_candidate": previous_candidate_test, "selected": leader_test, "paired_mcnemar": paired}, "iteration_count": len(list((run_dir / "evaluations").glob("*.json"))), "sealed_test_read_count": 1, "strict_holdout_used_for_training": False, "glyphs_used_for_training": False}
    dump_json(run_dir / "final-evaluation.json", final)

    if final_pass:
        runtime_dir = run_dir / "candidate-runtime"
        weights_dir = runtime_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        torch.save({key: value.detach().cpu() for key, value in leader["model"].state_dict().items()}, weights_dir / "projection.pt")
        historic_payload = torch.load(args.historical_weights / "prototypes.pt", map_location="cpu", weights_only=False)
        historic_payload["prototypes"] = leader["prototypes"].cpu()
        historic_payload["class_labels"] = torch.tensor(historical.labels, dtype=torch.long)
        torch.save(historic_payload, weights_dir / "prototypes.pt")
        shutil.copy2(BACKEND_ROOT / "codex_model/config.json", runtime_dir / "config.json")
        dump_json(runtime_dir / "export-manifest.json", {"schema_version": "external-strict-candidate-runtime.v1", "selected_iteration": leader["spec"]["iteration"], "selected_name": leader["spec"]["name"], "historical_untouched": True, "runtime_compatible": True, "final_evaluation": str((run_dir / "final-evaluation.json").resolve()), "projection_sha256": sha256_file(weights_dir / "projection.pt"), "prototypes_sha256": sha256_file(weights_dir / "prototypes.pt")})

    decision_lines.extend(["## Final sealed test", "", f"- Selected: {leader['spec']['name']}", f"- Historical top-1: {historical_test['top1']:.6f}", f"- Selected top-1: {leader_test['top1']:.6f}", f"- Delta: {leader_test['top1'] - historical_test['top1']:+.6f}", f"- Exact McNemar p: {paired['exact_p_value']:.6f}", f"- Final pass: {final_pass}", ""])
    (run_dir / "decision-log.md").write_text("\n".join(decision_lines), encoding="utf-8")
    dump_json(state_path, {"status": "completed", "run_id": run_dir.name, "mission": "elements-baseline-replacement", "started_at": started_at, "updated_at": utc_now(), "completed_at": utc_now(), "iteration": final["iteration_count"], "leader": leader["spec"]["name"], "pass": final_pass, "max_runtime_seconds": 21600})
    print(json.dumps(final, indent=2))
    return 0 if final_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
