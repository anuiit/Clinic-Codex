"""Shared test-process defaults.

Production keeps authentication enabled by default. Legacy and route
characterization tests run without user/session fixtures, so they opt out
explicitly for the test process and its subprocesses. Security tests override
or remove this variable when exercising the production default.
"""

from __future__ import annotations

import os


os.environ.setdefault("AUTH_REQUIRED", "false")
