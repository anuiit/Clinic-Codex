import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

import backend.codex_pipeline.scripts.precompute_embeddings as precompute
from backend.codex_pipeline.data.class_order import validate_metadata_class_subset
from backend.codex_pipeline.data.metadata import load_metadata

build_source_groups = precompute.build_source_groups
validate_snapshot_contract = precompute.validate_snapshot_contract


def _write_snapshot_fixture(tmp_path):
    relative_image = "Elements/0001-alpha/row-1.bmp"
    image_path = tmp_path / relative_image
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4), (10, 20, 30)).save(image_path, format="BMP")
    output_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {
                "image_path": relative_image,
                "row_id": "row-1",
                "dataset_split": "train",
                "class_label": 0,
                "element_name": "alpha",
                "source_group": "group-1",
                "output_sha256": output_sha,
            }
        ]
    ).to_csv(metadata_path, index=False)
    manifest = {
        "schema_version": "training-snapshot.v2",
        "snapshot_id": "snapshot-test",
        "content_sha256": "b" * 64,
        "ready_for_training": True,
        "class_order": ["alpha"],
        "row_count": 1,
        "rows": [
            {
                "row_id": "row-1",
                "dataset_split": "train",
                "class_label": 0,
                "class_name": "alpha",
                "source_group": "group-1",
                "output_path": relative_image,
                "output_sha256": output_sha,
            }
        ],
    }
    manifest_path = tmp_path / "snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elements = [{"path": relative_image, "sha256": output_sha}]
    checksums = {
        "schema_version": "training-snapshot-checksums.v1",
        "snapshot_id": "snapshot-test",
        "snapshot_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "metadata_csv_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "element_count": 1,
        "elements_sha256": precompute.canonical_json_sha(elements),
    }
    (tmp_path / "checksums.json").write_text(json.dumps(checksums), encoding="utf-8")
    return load_metadata(str(metadata_path)), metadata_path, manifest_path, image_path


def _write_backbone_manifest(tmp_path, *, weights_sha256=None, source_tree_sha256=None):
    repository = tmp_path / "dinov2"
    repository.mkdir()
    (repository / "hubconf.py").write_text("# local torch hub fixture\n", encoding="utf-8")
    (repository / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"pinned weights")
    manifest_path = tmp_path / "pin.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "dinov2-local-pin.v1",
                "backbone": "dinov2_vits14",
                "repository_path": str(repository),
                "weights_path": str(weights),
                "weights_sha256": weights_sha256 or precompute.sha256_file(weights),
                "source_tree_sha256": source_tree_sha256 or precompute.sha256_tree(repository),
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_build_source_groups_uses_page_coordinates_when_available() -> None:
    metadata = pd.DataFrame(
        [
            {"image_path": "Elements/0001-atl/10_11_12-a.bmp", "codex": "10", "folio": "11", "page": "12"},
            {"image_path": "Elements/0002-cafe/loose.bmp", "codex": None, "folio": None, "page": None},
        ]
    )

    assert build_source_groups(metadata) == [
        "page:10:11:12",
        "image:Elements/0002-cafe/loose.bmp",
    ]


def test_validate_metadata_class_subset_preserves_runtime_labels() -> None:
    metadata = pd.DataFrame(
        [
            {"class_label": 0, "element_name": "alpha"},
            {"class_label": 2, "element_name": "gamma"},
        ]
    )

    validate_metadata_class_subset(metadata, ["alpha", "beta", "gamma"])
    metadata.loc[1, "element_name"] = "beta"
    with pytest.raises(ValueError, match="differs from runtime contract"):
        validate_metadata_class_subset(metadata, ["alpha", "beta", "gamma"])


def test_build_source_groups_prefers_explicit_snapshot_groups() -> None:
    metadata = pd.DataFrame(
        [
            {
                "image_path": "Elements/0001-atl/a.bmp",
                "codex": "10",
                "folio": "11",
                "page": "12",
                "source_group": "content:abc",
            },
            {
                "image_path": "Elements/0002-cafe/b.bmp",
                "source_group": "content:def",
            },
        ]
    )

    assert build_source_groups(metadata) == ["content:abc", "content:def"]


def test_build_source_groups_rejects_empty_explicit_snapshot_groups() -> None:
    metadata = pd.DataFrame(
        [
            {"image_path": "Elements/0001-atl/a.bmp", "source_group": "content:abc"},
            {"image_path": "Elements/0002-cafe/b.bmp", "source_group": None},
        ]
    )

    try:
        build_source_groups(metadata)
    except ValueError as exc:
        assert "source_group is empty" in str(exc)
    else:
        raise AssertionError("empty explicit source groups must be rejected")


def test_validate_snapshot_contract_binds_manifest_metadata_and_class_order(tmp_path) -> None:
    metadata, metadata_path, manifest_path, _image_path = _write_snapshot_fixture(tmp_path)

    result = validate_snapshot_contract(
        metadata,
        metadata_csv=metadata_path,
        snapshot_manifest=manifest_path,
        runtime_class_order=["alpha"],
    )

    assert result["snapshot_id"] == "snapshot-test"
    with pytest.raises(ValueError, match="class order"):
        validate_snapshot_contract(
            metadata,
            metadata_csv=metadata_path,
            snapshot_manifest=manifest_path,
            runtime_class_order=["beta"],
        )


def test_validate_snapshot_contract_rejects_mutated_element_bytes(tmp_path) -> None:
    metadata, metadata_path, manifest_path, image_path = _write_snapshot_fixture(tmp_path)
    Image.new("RGB", (4, 4), (200, 20, 30)).save(image_path, format="BMP")

    with pytest.raises(ValueError, match="snapshot element checksum mismatch"):
        validate_snapshot_contract(
            metadata,
            metadata_csv=metadata_path,
            snapshot_manifest=manifest_path,
            runtime_class_order=["alpha"],
        )


def test_load_backbone_rejects_weights_hash_mismatch_before_torch_hub(tmp_path, monkeypatch) -> None:
    manifest_path = _write_backbone_manifest(tmp_path, weights_sha256="0" * 64)
    monkeypatch.setattr(
        precompute.torch.hub,
        "load",
        lambda *args, **kwargs: pytest.fail("torch.hub.load must not run before pin verification"),
    )

    with pytest.raises(ValueError, match="backbone weights checksum mismatch"):
        precompute.load_backbone("dinov2_vits14", precompute.torch.device("cpu"), manifest_path)


def test_load_backbone_rejects_source_tree_hash_mismatch_before_torch_hub(tmp_path, monkeypatch) -> None:
    manifest_path = _write_backbone_manifest(tmp_path, source_tree_sha256="0" * 64)
    monkeypatch.setattr(
        precompute.torch.hub,
        "load",
        lambda *args, **kwargs: pytest.fail("torch.hub.load must not run before pin verification"),
    )

    with pytest.raises(ValueError, match="backbone source tree checksum mismatch"):
        precompute.load_backbone("dinov2_vits14", precompute.torch.device("cpu"), manifest_path)


def test_sha256_tree_ignores_generated_python_caches(tmp_path: Path) -> None:
    repository = tmp_path / "dinov2"
    repository.mkdir()
    (repository / "hubconf.py").write_text("# stable\n", encoding="utf-8")
    expected = precompute.sha256_tree(repository)

    cache = repository / "__pycache__"
    cache.mkdir()
    (cache / "hubconf.cpython-311.pyc").write_bytes(b"generated")
    (repository / "module.pyc").write_bytes(b"generated")

    assert precompute.sha256_tree(repository) == expected
