# Worker 1 Linux disposable install/dev verification

Status: PASS (Linux lane)

Source commit tested: `142edfc61bd9b6b917b2c5eaa5e65b130a9486e1` (detached HEAD archive copy).
Disposable temp root: `/tmp/clinic-codex-new-user-linux-20260526T121132Z-worker1`.
Cleanup: marker + canonical prefix checked; root removed.

## Verification matrix

- PASS: `bash -n scripts/install.sh scripts/run-dev.sh` in disposable archive copy.
- PASS: `bash scripts/install.sh` first run created `backend/.venv/bin/python`, non-empty `frontend/node_modules`, and `backend/codex_model/weights/prototypes.pt`.
- PASS: Python import sanity for `flask, torch, torchvision, mobile_sam, segment_anything, albumentations, timm`.
- PASS: `run-dev.sh` smoke on alternate ports `44073/47479`; `/classes` and `/` answered; log printed matching backend/frontend URLs; listeners stopped after interrupt.
- PASS: installer idempotency second run reused venv/node_modules/prototypes.
- PASS: second `run-dev.sh` smoke on alternate ports `59167/44957`; `/classes` and `/` answered; log printed matching backend/frontend URLs; listeners stopped after interrupt.
- PASS: no listeners remained on tested ports after shutdown.

## Evidence files

- `linux-lane-20260526T121132Z.log`
- `linux-run-dev-first-20260526T121132Z.log`
- `linux-run-dev-second-20260526T121132Z.log`
- `cleanup-linux-20260526T121132Z.log`
- `linux-summary-20260526T121132Z.md`

## Notes

- Earlier harness attempts are preserved as evidence logs but superseded by the PASS run above; one prior smoke harness killed `run-dev.sh` before waiting for final URL log lines even though services answered.
- `npm audit` reported 2 moderate vulnerabilities during install; not part of this install/dev script acceptance lane.
