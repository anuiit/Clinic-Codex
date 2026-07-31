from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "generate_model_comparison_html.py"


def _fixture(tmp_path: Path) -> tuple[Path, list[dict[str, object]]]:
    image = tmp_path / "sample.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    rows: list[dict[str, object]] = [
        {
            "path": image.name,
            "truth": "tooth",
            "historical": {
                "pred": "tooth",
                "top3": ["tooth", "implant", "crown"],
                "correct": True,
            },
            "v5": {
                "pred": "implant",
                "top3": ["implant", "tooth", "crown"],
                "correct": False,
            },
            "augmented": {
                "pred": "tooth",
                "top3": ["tooth", "crown", "implant"],
                "correct": True,
            },
        },
        {
            "path": "missing.png",
            "truth": "crown",
            "historical": {"pred": "implant", "top3": ["implant"], "correct": False},
            "v5": {"pred": "crown", "top3": ["crown"], "correct": True},
        },
    ]
    input_path = tmp_path / "rows.json"
    input_path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return input_path, rows


def test_generates_autonomous_comparison_report(tmp_path: Path) -> None:
    input_path, _ = _fixture(tmp_path)
    output_path = tmp_path / "report.html"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(input_path),
            str(output_path),
            "--max-images",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = output_path.read_text(encoding="utf-8")
    encoded_image = base64.b64encode((tmp_path / "sample.png").read_bytes()).decode("ascii")
    assert "Wrote" in completed.stdout
    assert "<!doctype html>" in report
    assert f"data:image/png;base64,{encoded_image}" in report
    assert "Comparaison visuelle des modèles" in report
    assert "Historique" in report
    assert "V5" in report
    assert "Augmenté" in report
    assert "Top-1&nbsp;: 50.0%" in report
    assert "Chemin d’image absent" not in report
    assert "Image introuvable" in report
    assert 'id="modelFilter"' in report
    assert 'id="resultFilter"' in report
    assert 'id="classFilter"' in report
    assert 'id="search"' in report
    assert 'event.key === "ArrowLeft"' in report
    assert 'event.key === "ArrowRight"' in report
    assert "https://" not in report


def test_missing_augmented_predictions_are_supported_and_limit_is_applied(
    tmp_path: Path,
) -> None:
    input_path, rows = _fixture(tmp_path)
    rows[0].pop("augmented")
    input_path.write_text(json.dumps(rows), encoding="utf-8")
    output_path = tmp_path / "limited.html"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(input_path),
            str(output_path),
            "--max-images",
            "1",
        ],
        check=True,
    )

    report = output_path.read_text(encoding="utf-8")
    assert "1 images intégrées" in report
    assert "Augmenté" not in report
    assert "missing.png" not in report
    assert "Historique" in report
    assert "V5" in report


def test_rejects_invalid_prediction_payload(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"rows": "not-a-list"}), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(invalid), str(tmp_path / "out.html")],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "must be a list" in completed.stderr


@pytest.mark.parametrize("max_images", [0, -1])
def test_rejects_non_positive_max_images(tmp_path: Path, max_images: int) -> None:
    input_path, _ = _fixture(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(input_path),
            str(tmp_path / "out.html"),
            "--max-images",
            str(max_images),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "max_images must be at least 1" in completed.stderr
