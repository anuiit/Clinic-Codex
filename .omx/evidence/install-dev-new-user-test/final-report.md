# Install/dev new-user verification final report

Generated: 2026-05-26T12:18Z

## Overall classification

**SCRIPT FAILURE for native Windows install/runtime validation.**

Linux new-user install/dev passed end-to-end in a disposable archive copy. Native Windows PowerShell parsing and preflight passed, and Windows `py -3.11` was proven available, but `scripts/install.ps1` failed at Stage 0 with `Need Python 3.10 or 3.11.` Runtime smoke on Windows was therefore not attempted, and this report makes **no Windows runtime success claim**.

## Source commits and evidence scope

- Linux lane tested source commit: `142edfc61bd9b6b917b2c5eaa5e65b130a9486e1`.
- Windows lane correction evidence source commit: `dc0ea43ac5d1b52c13bab8139822a042bfc8b791`.
- Replacement final audit supersedes failed task 6, which was failed only because the leader gate had not yet been satisfied.
- Evidence root: `/home/sina/omxpro/clinic-codex/.omx/evidence/install-dev-new-user-test/`.

## Command / lane matrix

| Lane | Evidence | Classification | Notes |
| --- | --- | --- | --- |
| Linux syntax/install/import | `worker1-linux-report.md`, `linux-lane-20260526T121132Z.log`, `linux-summary-20260526T121132Z.md` | PASS | `bash -n` passed; `bash scripts/install.sh` created venv, frontend deps, and prototypes; critical Python imports passed. |
| Linux run-dev smoke #1 | `linux-run-dev-first-20260526T121132Z.log` | PASS | Alternate ports `44073/47479`; backend `/classes` and frontend `/` answered; printed URLs matched; listeners stopped. |
| Linux idempotency + smoke #2 | `linux-run-dev-second-20260526T121132Z.log` | PASS | Reinstall reused existing venv/node_modules/prototypes; second smoke passed on `59167/44957`; listeners stopped. |
| Linux cleanup | `cleanup-linux-20260526T121132Z.log` | PASS | Temp root `/tmp/clinic-codex-new-user-linux-20260526T121132Z-worker1` removed after marker + canonical prefix check. |
| Marker cleanup guard | `worker-2-marker-files-20260526T120740Z.md` | PASS | Marker-only evidence; useful for cleanup guard proof, not Windows runtime proof. |
| Native Windows parser/preflight | `worker-2-windows-lane-20260526T121215Z.log`, `.md` summary | PASS | Native Windows PowerShell 5.1 parser check passed for `scripts/install.ps1` and `scripts/run-dev.ps1`; versions recorded without aborting. |
| Native Windows install | `worker-2-windows-install-20260526T121215Z.log`, `worker-2-windows-python-diagnostic-20260526T121215Z.log` | SCRIPT FAILURE | `py -3.11 --version` and direct diagnostic reported Python 3.11.4 / 3.11, but installer failed Stage 0 claiming Python 3.10/3.11 was missing. |
| Native Windows run-dev smoke | `worker-2-windows-run-dev-20260526T121215Z.log` | NOT RUN after script failure | Empty by design because install failed. No Windows runtime success claim. |
| Final no-lingering audit | latest `final-audit-*.log` | PASS | Common/default/test ports free; no matching disposable temp roots or app launcher processes found. |

## Temp roots

- Linux: `/tmp/clinic-codex-new-user-linux-20260526T121132Z-worker1` — marker/canonical cleanup verified; removed.
- Windows: `/mnt/c/Temp/clinic-codex-new-user-win-worker2-runtime-20260526T121215Z` / `C:\Temp\clinic-codex-new-user-win-worker2-runtime-20260526T121215Z` — `.clinic-codex-temp-root` plus approved prefix checked; deletion verified from Linux and Windows views.
- Final audit found no remaining `clinic-codex-new-user-*` temp roots under `/tmp` or `/mnt/c/Temp`.

## No-lingering proof

Final audit log: latest `final-audit-*.log`.

- PASS: no listeners on default ports `7117/7118`.
- PASS: no listeners on Linux smoke ports `44073/47479` or `59167/44957`.
- PASS: no relevant lingering `flask --app`, Vite dev server, `run-dev.sh`, `run-dev.ps1`, or disposable temp-root processes matched.

## Follow-up bugs

1. **Fix Windows Python detection in `scripts/install.ps1`.** Evidence proves `py -3.11` works under native Windows PowerShell 5.1, but installer Stage 0 did not accept it.
2. **Do not count marker-only cleanup evidence as Windows runtime evidence.** Worker-2 task 5 is cleanup guard evidence only; task 8 is the actual native Windows verifier lane.
3. **Preserve Windows runtime gating.** A Windows runtime PASS must require both successful `install.ps1` and successful `run-dev.ps1` smoke. Current Windows lane stops correctly after install failure.
4. **Optional harness improvement:** earlier Linux logs from `20260526T120809Z` show a premature harness failure because it killed `run-dev.sh` before URL log checks. The later `20260526T121132Z` Linux PASS supersedes it.

## Final verdict

- Linux new-user install/dev: **PASS**.
- Native Windows install/dev: **SCRIPT FAILURE**.
- Cleanup/no-lingering: **PASS**.
- Overall release readiness for cross-platform new-user install/dev: **FAIL until Windows Python detection is fixed and Windows install + smoke pass.**
