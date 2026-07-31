"""Small server-side identity, session, and authorization layer.

The application deliberately keeps this dependency-free: users and revocable
sessions live in the configured SQLite database, while cookies contain only an
opaque random session identifier.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

from flask import current_app, g, jsonify, request
from werkzeug.security import check_password_hash, generate_password_hash

_DUMMY_PASSWORD_HASH = generate_password_hash("not-a-real-password")

F = TypeVar("F", bound=Callable[..., object])

ROLES = {"contributor", "reviewer", "ml_operator", "org_admin"}
ROLE_PERMISSIONS = {
    "contributor": {"analysis.submit"},
    "reviewer": {"annotation.queue.read", "annotation.review"},
    "ml_operator": {"training.read", "training.run"},
    "org_admin": {"analysis.submit", "annotation.queue.read", "annotation.review", "training.read", "training.run", "member.manage"},
}
COOKIE_NAME = "clinic_session"
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 256
_EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


class BootstrapValidationError(ValueError):
    def __init__(self, error_code: str, message: str):
        super().__init__(message)
        self.error_code = error_code


class BootstrapUnavailableError(ValueError):
    def __init__(self):
        super().__init__("bootstrap unavailable")


def validate_bootstrap_credentials(email: object, password: object) -> tuple[str, str]:
    if not isinstance(email, str):
        raise BootstrapValidationError("INVALID_EMAIL", "A valid email address is required.")
    normalized_email = email.strip().lower()
    if (
        not normalized_email
        or len(normalized_email) > 254
        or not _EMAIL_PATTERN.fullmatch(normalized_email)
    ):
        raise BootstrapValidationError("INVALID_EMAIL", "A valid email address is required.")
    if not isinstance(password, str) or not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
        raise BootstrapValidationError(
            "INVALID_PASSWORD",
            f"Password must contain between {MIN_PASSWORD_LENGTH} and {MAX_PASSWORD_LENGTH} characters.",
        )
    return normalized_email, password


class LoginAttemptLimiter:
    """In-memory guard for interactive login attempts; reset on process restart."""

    def __init__(self, *, limit: int = 5, window_seconds: int = 300):
        self.limit = limit
        self.window_seconds = window_seconds
        self._attempts: dict[tuple[str, str], list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, email: str, remote_addr: str | None) -> bool:
        key = (email.strip().lower(), remote_addr or "")
        now = time.monotonic()
        with self._lock:
            attempts = [item for item in self._attempts.get(key, []) if now - item < self.window_seconds]
            self._attempts[key] = attempts
            return len(attempts) < self.limit

    def record_failure(self, email: str, remote_addr: str | None) -> bool:
        key = (email.strip().lower(), remote_addr or "")
        now = time.monotonic()
        with self._lock:
            attempts = [item for item in self._attempts.get(key, []) if now - item < self.window_seconds]
            attempts.append(now)
            self._attempts[key] = attempts
            return len(attempts) < self.limit

    def clear(self, email: str, remote_addr: str | None) -> None:
        with self._lock:
            self._attempts.pop((email.strip().lower(), remote_addr or ""), None)
CSRF_HEADER = "X-CSRF-Token"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat()


class AuthStore:
    def __init__(self, database_path: Path, *, session_ttl_hours: int = 24 * 7):
        self.database_path = Path(database_path)
        self.session_ttl_hours = session_ttl_hours
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id),
                    csrf_hash TEXT NOT NULL,
                    csrf_token TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    revoked_at TEXT
                );
                CREATE INDEX IF NOT EXISTS sessions_user_idx ON sessions(user_id);
                """
            )

            try:
                conn.execute("ALTER TABLE sessions ADD COLUMN csrf_token TEXT")
            except sqlite3.OperationalError:
                pass
    def ensure_bootstrap_user(self, email: str, password: str, role: str) -> None:
        if role not in ROLES:
            raise RuntimeError("AUTH_BOOTSTRAP_ROLE must be a supported role")
        if not email or not password:
            return
        normalized_email, validated_password = validate_bootstrap_credentials(email, password)
        try:
            self._create_first_user(normalized_email, validated_password, role)
        except BootstrapUnavailableError:
            # Environment bootstrap is only a first-run convenience. It must
            # never add a privileged account to an initialized database.
            return

    def bootstrap_available(self) -> bool:
        with self._connection() as conn:
            row = conn.execute("SELECT 1 FROM users LIMIT 1").fetchone()
        return row is None

    def create_first_admin(self, email: object, password: object) -> dict[str, Any]:
        normalized_email, validated_password = validate_bootstrap_credentials(email, password)
        return self._create_first_user(normalized_email, validated_password, "org_admin")

    def _create_first_user(self, email: str, password: str, role: str) -> dict[str, Any]:
        user = {
            "id": secrets.token_urlsafe(18),
            "email": email,
            "role": role,
            "active": True,
            "created_at": _iso(_now()),
        }
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None:
                raise BootstrapUnavailableError()
            conn.execute(
                "INSERT INTO users(id, email, password_hash, role, active, created_at) VALUES (?, ?, ?, ?, 1, ?)",
                (
                    user["id"],
                    user["email"],
                    generate_password_hash(password),
                    role,
                    user["created_at"],
                ),
            )
        return {
            "id": user["id"],
            "email": user["email"],
            "role": role,
            "roles": [role],
            "permissions": sorted(ROLE_PERMISSIONS[role]),
        }

    def create_user(self, email: str, password: str, role: str) -> dict[str, Any]:
        if role not in ROLES:
            raise ValueError("unsupported role")
        if not email.strip() or not password:
            raise ValueError("email and password are required")
        user = {"id": secrets.token_urlsafe(18), "email": email.strip(), "role": role, "active": True, "created_at": _iso(_now())}
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO users(id, email, password_hash, role, active, created_at) VALUES (?, ?, ?, ?, 1, ?)",
                (user["id"], user["email"], generate_password_hash(password), role, user["created_at"]),
            )
        return user

    def authenticate(self, email: str, password: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE email = ? AND active = 1", (email.strip(),)).fetchone()
        if row is None:
            check_password_hash(_DUMMY_PASSWORD_HASH, password)
            return None
        if not check_password_hash(row["password_hash"], password):
            return None
        return _public_user(row)

    def create_session(self, user_id: str) -> tuple[str, str]:
        session_id, csrf_token = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        expires = _now() + timedelta(hours=self.session_ttl_hours)
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO sessions(id_hash, user_id, csrf_hash, csrf_token, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
                (_digest(session_id), user_id, _digest(csrf_token), csrf_token, _iso(_now()), _iso(expires)),
            )
        return session_id, csrf_token

    def session_user(self, session_id: str | None) -> tuple[dict[str, Any], str] | None:
        if not session_id:
            return None
        with self._connection() as conn:
            row = conn.execute(
                """SELECT u.*, s.csrf_hash FROM sessions s JOIN users u ON u.id=s.user_id
                   WHERE s.id_hash=? AND s.revoked_at IS NULL AND s.expires_at > ? AND u.active=1""",
                (_digest(session_id), _iso(_now())),
            ).fetchone()
        return (_public_user(row), row["csrf_hash"]) if row is not None else None

    def csrf_token_for_session(self, session_id: str | None) -> str | None:
        """Return the stable CSRF token for a live session, recovering legacy rows once."""
        if not session_id:
            return None
        with self._connection() as conn:
            row = conn.execute(
                "SELECT csrf_token FROM sessions WHERE id_hash=? AND revoked_at IS NULL AND expires_at > ?",
                (_digest(session_id), _iso(_now())),
            ).fetchone()
        if row is None:
            return None
        if row["csrf_token"]:
            return row["csrf_token"]
        return self.rotate_csrf_token(session_id)
    def rotate_csrf_token(self, session_id: str | None) -> str | None:
        """Issue a replacement CSRF token for the current valid server-side session."""
        if not session_id:
            return None
        token = secrets.token_urlsafe(32)
        with self._connection() as conn:
            cursor = conn.execute(
                "UPDATE sessions SET csrf_hash=?, csrf_token=? WHERE id_hash=? AND revoked_at IS NULL AND expires_at > ?",
                (_digest(token), token, _digest(session_id), _iso(_now())),
            )
        return token if cursor.rowcount == 1 else None
    def revoke_session(self, session_id: str | None) -> None:
        if not session_id:
            return
        with self._connection() as conn:
            conn.execute("UPDATE sessions SET revoked_at=? WHERE id_hash=?", (_iso(_now()), _digest(session_id)))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _public_user(row: sqlite3.Row) -> dict[str, Any]:
    role = row["role"]
    return {"id": row["id"], "email": row["email"], "role": role, "roles": [role], "permissions": sorted(ROLE_PERMISSIONS[role])}


def auth_enabled() -> bool:
    return bool(current_app.config["CLINIC_SETTINGS"].authentication_enabled)


def current_user() -> dict[str, Any] | None:
    return getattr(g, "clinic_user", None)


def load_current_user() -> None:
    if not auth_enabled():
        return
    session = current_app.extensions["clinic_auth_store"].session_user(request.cookies.get(COOKIE_NAME))
    if session is not None:
        g.clinic_user, g.clinic_csrf_hash = session


def require_permission(permission: str):
    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not auth_enabled():
                return fn(*args, **kwargs)
            user = current_user()
            if user is None:
                return jsonify({"status": "error", "error_code": "AUTH_REQUIRED", "error": "authentication required"}), 401
            if permission not in user["permissions"]:
                return jsonify({"status": "error", "error_code": "PERMISSION_DENIED", "error": "permission denied"}), 403
            return fn(*args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return decorator


def require_csrf(fn: F) -> F:
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if auth_enabled() and request.method not in {"GET", "HEAD", "OPTIONS"}:
            token = request.headers.get(CSRF_HEADER, "")
            expected = getattr(g, "clinic_csrf_hash", "")
            if not token or not expected or not hmac.compare_digest(_digest(token), expected):
                return jsonify({"status": "error", "error_code": "CSRF_INVALID", "error": "valid X-CSRF-Token required"}), 403
        return fn(*args, **kwargs)
    return wrapper  # type: ignore[return-value]
