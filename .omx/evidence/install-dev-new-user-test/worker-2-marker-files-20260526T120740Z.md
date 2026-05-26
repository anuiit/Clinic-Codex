# Worker 2 Task 5 — marker files

Result: PASS

Marker filename used: `.clinic-codex-temp-root`

Evidence log: `worker-2-marker-files-20260526T120740Z.log`

Checks:
- PASS Linux root used allowed prefix `/tmp/clinic-codex-new-user-linux-*`.
- PASS Linux cleanup was gated on marker file existence and canonical prefix check.
- PASS Windows-accessible root used allowed prefix when /mnt/c was available; otherwise logged as unavailable.
- PASS All worker-created marker temp roots were removed after evidence capture.

Stop condition honored: no cleanup target was deleted unless it matched the allowed prefix and contained `.clinic-codex-temp-root`.
