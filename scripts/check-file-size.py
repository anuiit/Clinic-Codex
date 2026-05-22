#!/usr/bin/env python3
"""Warning-only source file size guard for the maintainability roadmap.

Phase 0 intentionally exits 0 even when warnings are emitted. Later roadmap
phases can make this check blocking after oversized files have been refactored.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WARNING_ONLY = True

CHECKS = [
    (ROOT / "frontend" / "src", {".ts", ".tsx"}, 500, "frontend source"),
    (ROOT / "backend" / "examples" / "flask_api.py", {".py"}, 250, "backend route/server"),
    (ROOT / "backend" / "app" / "routes", {".py"}, 250, "backend route/server"),
]


def line_count(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
    except OSError as exc:
        print(f"WARNING: could not read {path.relative_to(ROOT)}: {exc}")
        return 0


def main() -> int:
    warnings: list[str] = []
    for base, suffixes, threshold, label in CHECKS:
        if not base.exists():
            continue

        candidates = [base] if base.is_file() else sorted(base.rglob("*"))
        for path in candidates:
            if not path.is_file() or path.suffix not in suffixes:
                continue
            if ".test." in path.name:
                continue
            count = line_count(path)
            if count > threshold:
                warnings.append(
                    f"{path.relative_to(ROOT)} has {count} lines "
                    f"(>{threshold} {label} Phase 0 warning threshold)"
                )

    if warnings:
        print("File-size guard: WARNING-ONLY in Phase 0")
        for warning in warnings:
            print(f"WARNING: {warning}")
    else:
        print("File-size guard: no files exceed Phase 0 warning thresholds")

    print("File-size guard result: pass (warning-only; exits 0 in Phase 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
