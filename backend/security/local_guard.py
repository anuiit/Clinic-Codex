"""Local-only request helpers for Phase 1 admin surfaces.

These helpers are not an authentication system. They intentionally support the
current direct local binding model only and do not trust reverse-proxy headers.
"""
from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from functools import wraps
from typing import Callable, TypeVar
from urllib.parse import urlparse

from flask import jsonify, request

F = TypeVar("F", bound=Callable[..., object])


def _host_without_port(value: str | None) -> str:
    if not value:
        return ""
    host = value.strip().lower()
    if host.startswith("[") and "]" in host:
        return host[1 : host.index("]")]
    if ":" in host and host.count(":") == 1:
        return host.split(":", 1)[0]
    return host


def is_loopback_address(value: str | None) -> bool:
    if not value:
        return False
    host = _host_without_port(value)
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def is_local_origin(origin: str | None) -> bool:
    if not origin:
        return True
    parsed = urlparse(origin)
    return is_loopback_address(parsed.hostname)


@dataclass(frozen=True)
class RequestLaunchContext:
    remote_addr: str | None
    host: str | None
    origin: str | None


def local_request_denial_reasons(context: RequestLaunchContext) -> list[str]:
    reasons: list[str] = []
    if not is_loopback_address(context.remote_addr):
        reasons.append("non_loopback_remote_addr")
    if not is_loopback_address(context.host):
        reasons.append("nonlocal_host")
    if not is_local_origin(context.origin):
        reasons.append("nonlocal_origin")
    return reasons


def request_launch_context() -> RequestLaunchContext:
    return RequestLaunchContext(
        remote_addr=request.remote_addr,
        host=request.host,
        origin=request.headers.get("Origin"),
    )


def require_local_request(fn: F) -> F:
    @wraps(fn)
    def wrapper(*args, **kwargs):
        reasons = local_request_denial_reasons(request_launch_context())
        if reasons:
            return (
                jsonify(
                    {
                        "status": "error",
                        "error_code": "LOCAL_ONLY_FORBIDDEN",
                        "error": "local-only admin endpoint",
                        "message": "This admin endpoint is only available from a local loopback browser or tool.",
                        "reasons": reasons,
                    }
                ),
                403,
            )
        return fn(*args, **kwargs)

    setattr(wrapper, "__clinic_local_required__", True)
    return wrapper  # type: ignore[return-value]
