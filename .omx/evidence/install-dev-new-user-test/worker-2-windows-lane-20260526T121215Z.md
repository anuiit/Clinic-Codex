# Worker 2 Windows lane correction

Classification: SCRIPT FAILURE — native Windows Python 3.11 is available via `py -3.11`, but `scripts/install.ps1` failed during Stage 0 with "Need Python 3.10 or 3.11".

Source commit: dc0ea43ac5d1b52c13bab8139822a042bfc8b791

Temp root used and removed: `/mnt/c/Temp/clinic-codex-new-user-win-worker2-runtime-20260526T121215Z` / `C:\Temp\clinic-codex-new-user-win-worker2-runtime-20260526T121215Z`

Evidence:
- Parser/preflight/cleanup log: `worker-2-windows-lane-20260526T121215Z.log`
- Install log: `worker-2-windows-install-20260526T121215Z.log`
- Python diagnostic: `worker-2-windows-python-diagnostic-20260526T121215Z.log`
- Run-dev log: `worker-2-windows-run-dev-20260526T121215Z.log` (empty because install failed; runtime not attempted)

Key facts:
- PASS native Windows PowerShell 5.1 parser check for `scripts/install.ps1` and `scripts/run-dev.ps1`.
- PASS preflight recorded PowerShell, `py -3.11`, `py -3.10`, `python`, `node`, and `npm` outputs.
- SCRIPT FAILURE: `py -3.11 --version` reported Python 3.11.4 and direct diagnostic confirmed `py -3.11 -c` reports 3.11, but installer failed to detect it.
- Runtime not attempted because install did not succeed; no Windows runtime success is claimed.
- PASS marker-guarded cleanup: approved prefix + `.clinic-codex-temp-root` checked before deletion; deletion verified from Linux and Windows views.
