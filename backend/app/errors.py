"""Shared API error helpers."""
from __future__ import annotations

import traceback
import uuid
from dataclasses import dataclass
from typing import Any

from flask import jsonify

try:  # Support both repo-root (`backend.*`) and backend-root (`services.*`) imports.
    from backend.services.annotation_storage import (
        AnnotationDiskFullError,
        AnnotationPermissionError,
        AnnotationStorageError,
    )
except ImportError:  # pragma: no cover - compatibility path
    from services.annotation_storage import (  # type: ignore
        AnnotationDiskFullError,
        AnnotationPermissionError,
        AnnotationStorageError,
    )


@dataclass
class ApiError(Exception):
    code: str
    message: str
    status_code: int = 400
    payload: dict[str, Any] | None = None

    def to_response(self):
        body = {"error": {"code": self.code, "message": self.message}}
        if self.payload:
            body["error"].update(self.payload)
        return jsonify(body), self.status_code


@dataclass
class ModelAssetUnavailable(Exception):
    """Raised when a local ML asset is missing and request handling cannot proceed."""

    asset: str
    path: str
    hint: str

    def to_response(self):
        return (
            jsonify(
                {
                    "error": {
                        "code": "MODEL_ASSET_UNAVAILABLE",
                        "message": f"Required model asset is unavailable: {self.asset}",
                        "asset": self.asset,
                        "path": self.path,
                        "hint": self.hint,
                    }
                }
            ),
            503,
        )


def annotation_error_response(exc: Exception):
    if isinstance(exc, ValueError):
        message = str(exc)
        return jsonify({"status": "error", "error_code": "VALIDATION_ERROR", "message": message, "error": message}), 400
    if isinstance(exc, AnnotationPermissionError):
        return jsonify(
            {
                "status": "error",
                "error_code": "PERMISSION_DENIED",
                "message": "Droits insuffisants sur le dossier annotations",
                "hint": "Vérifiez les permissions du dossier backend/annotations/ ou lancez l'application avec un compte ayant accès en écriture.",
            }
        ), 409
    if isinstance(exc, AnnotationDiskFullError):
        return jsonify(
            {
                "status": "error",
                "error_code": "DISK_FULL",
                "message": "Espace disque insuffisant",
                "hint": "Libérez de l'espace puis réessayez.",
            }
        ), 507
    if isinstance(exc, AnnotationStorageError):
        return jsonify(
            {
                "status": "error",
                "error_code": "STORAGE_ERROR",
                "message": str(exc),
                "hint": None,
            }
        ), 500

    trace_id = uuid.uuid4().hex[:8]
    traceback.print_exc()
    return jsonify(
        {
            "status": "error",
            "error_code": "INTERNAL_ERROR",
            "message": f"Erreur interne du serveur (id={trace_id})",
            "hint": "Contactez le support avec cet identifiant.",
            "trace_id": trace_id,
        }
    ), 500
