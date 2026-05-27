#!/usr/bin/env python3
"""Source file size guard for the maintainability roadmap.

Phase 3 makes frontend source files blocking at the existing 500-line threshold
after the ImageBBoxStage migration brought scoped files under the limit. Backend
route/server checks remain warning-only so unrelated backend work is not blocked
by this frontend migration phase.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKS = [
    (ROOT / "frontend" / "src", {".ts", ".tsx"}, 500, "frontend source", False),
    (ROOT / "backend" / "examples" / "flask_api.py", {".py"}, 250, "backend route/server", True),
    (ROOT / "backend" / "app" / "routes", {".py"}, 250, "backend route/server", True),
]


def line_count(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
    except OSError as exc:
        print(f"WARNING: could not read {path.relative_to(ROOT)}: {exc}")
        return 0


def main() -> int:
    warnings: list[str] = []
    failures: list[str] = []
    for base, suffixes, threshold, label, warning_only in CHECKS:
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
                message = (
                    f"{path.relative_to(ROOT)} has {count} lines "
                    f"(>{threshold} {label} Phase 3 threshold)"
                )
                if warning_only:
                    warnings.append(message)
                else:
                    failures.append(message)

    if failures:
        print("File-size guard: FAIL")
        for failure in failures:
            print(f"ERROR: {failure}")

    if warnings:
        print("File-size guard: warnings")
        for warning in warnings:
            print(f"WARNING: {warning}")

    if failures:
        print("File-size guard result: fail (frontend source threshold is blocking)")
        return 1

    if not warnings:
        print("File-size guard: no files exceed Phase 3 thresholds")

    print("File-size guard result: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
