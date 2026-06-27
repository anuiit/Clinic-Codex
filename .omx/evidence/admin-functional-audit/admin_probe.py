"""End-to-end functional probe of the admin dashboard endpoints (worker-2 audit).

Boots the real Flask app via create_app() against a temp backend_root and
exercises every admin capability through the HTTP layer (test client), then
verifies the training launch guard transitions. No mocks.
"""
from __future__ import annotations

import base64
import io
import json
import tempfile
from pathlib import Path

from PIL import Image

from backend.app.config import Settings
from backend.app.factory import create_app


def png_data_url(size=(16, 16)):
    buf = io.BytesIO()
    Image.new("RGB", size, color=(200, 180, 160)).save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def payload(analysis_id):
    return {
        "analysis_id": analysis_id,
        "image_data_url": png_data_url(),
        "annotations": [
            {"index": 0, "class_name": "atl", "bbox": [0, 0, 6, 6]},
            {"index": 1, "class_name": "calli", "bbox": [6, 6, 6, 6]},
        ],
    }


LOOPBACK = {"REMOTE_ADDR": "127.0.0.1"}
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))


with tempfile.TemporaryDirectory() as td:
    backend_root = Path(td)
    settings = Settings(backend_root=backend_root, testing=True)
    app = create_app(settings=settings)
    c = app.test_client()

    # 1. Submit annotation -> appears in admin queue as pending
    save = c.post("/save-annotation", json=payload("probe-1"))
    check("save-annotation 200", save.status_code == 200, str(save.status_code))

    q = c.get("/admin/annotations")
    check("GET /admin/annotations 200", q.status_code == 200)
    qj = q.get_json()
    check("queue counts total=2", qj["counts"]["total"] == 2, json.dumps(qj["counts"]))
    check("queue starts all-pending", qj["counts"]["pending"] == 2 and qj["counts"]["trainable"] == 0)
    check("queue local_only warning present", bool(qj.get("warning")))

    # 2. Approve element 0 -> becomes trainable (crop exists + fresh fingerprint)
    appr = c.post("/admin/annotations/probe-1/0/review", json={"status": "approved"})
    check("approve element 0 -> 200", appr.status_code == 200)
    el0 = appr.get_json()["element"]
    check("approved element trainable", el0["review_status"] == "approved" and el0["trainable"] is True,
          f"status={el0['review_status']} trainable={el0['trainable']}")

    # 3. Reject element 1 -> excluded from training
    rej = c.post("/admin/annotations/probe-1/1/review", json={"status": "rejected"})
    el1 = rej.get_json()["element"]
    check("reject element 1 not trainable", el1["review_status"] == "rejected" and el1["trainable"] is False)

    # 4. Modify element 0 (new class + bbox) -> regenerates crop, returns to pending
    before_crop = el0["crop_path"]
    mod = c.post("/admin/annotations/probe-1/0/modify",
                 json={"class_name": "new-atl", "bbox": [1, 1, 5, 5]})
    check("modify element 0 -> 200", mod.status_code == 200, str(mod.status_code))
    mel = mod.get_json()["element"]
    check("modified element back to pending", mel["review_status"] == "pending" and mel["trainable"] is False)
    check("modified crop regenerated (new path, exists)",
          mel["crop_path"] != before_crop and mel["crop_exists"] is True)
    check("modified class persisted", mel["class_name"] == "new-atl", mel["class_name"])
    check("old crop removed", not Path(before_crop).is_file())

    # 5. Save & approve in one modify intent
    mod2 = c.post("/admin/annotations/probe-1/0/modify",
                  json={"class_name": "new-atl", "bbox": [1, 1, 5, 5], "approve_after_save": True})
    check("save&approve -> approved+trainable",
          mod2.get_json()["element"]["review_status"] == "approved" and mod2.get_json()["element"]["trainable"] is True)

    # 6. Crop + image media endpoints serve files
    img = c.get("/admin/annotations/probe-1/image")
    crop = c.get("/admin/annotations/probe-1/0/crop")
    check("image media endpoint 200", img.status_code == 200)
    check("crop media endpoint 200", crop.status_code == 200)

    # 7. Invalid inputs rejected
    bad_status = c.post("/admin/annotations/probe-1/0/review", json={"status": "bogus"})
    check("invalid review status -> 400", bad_status.status_code == 400)
    missing = c.post("/admin/annotations/probe-1/9/review", json={"status": "approved"})
    check("missing element -> 404", missing.status_code == 404)

    # 8. Training summary: DISABLED BY DEFAULT (intentional, not a defect)
    ts = c.get("/admin/training/summary", environ_overrides=LOOPBACK)
    check("training summary 200", ts.status_code == 200)
    tsj = ts.get_json()
    check("launch disabled by default", tsj["launch_allowed_for_request"] is False)
    check("reason is disabled_by_default",
          any(r.startswith("disabled_by_default:") for r in tsj["launch_disabled_reasons"]),
          json.dumps(tsj["launch_disabled_reasons"]))
    check("summary reports trainable count", tsj["data"]["trainable"] == 1, str(tsj["data"]["trainable"]))

    # 9. POST training job while disabled -> 403 forbidden (guard holds)
    start = c.post("/admin/training/jobs", json={"dry_run": True}, environ_overrides=LOOPBACK)
    check("start job blocked while disabled -> 403", start.status_code == 403, str(start.status_code))


# 10. With ENABLE flag ON, disabled_by_default reason disappears (proves the gate)
with tempfile.TemporaryDirectory() as td2:
    settings2 = Settings(backend_root=Path(td2), testing=True, enable_admin_training_jobs=True)
    app2 = create_app(settings=settings2)
    c2 = app2.test_client()
    c2.post("/save-annotation", json=payload("probe-2"))
    ts2 = c2.get("/admin/training/summary", environ_overrides=LOOPBACK).get_json()
    reasons = ts2["launch_disabled_reasons"]
    check("flag ON removes disabled_by_default",
          not any(r.startswith("disabled_by_default:") for r in reasons), json.dumps(reasons))
    # tmp backend_root has no scripts/retrain.sh, so the only remaining blocker is the missing script
    check("flag ON: only remaining blocker is missing local retrain script",
          all(r.startswith("missing_retrain_script") for r in reasons) or not reasons,
          json.dumps(reasons))

# 11. The real repo ships scripts/retrain.sh so a real local+loopback run would be launchable
repo_script = Path("/home/sina/omxpro/clinic-codex/scripts/retrain.sh")
check("repo ships scripts/retrain.sh (real local launch path exists)", repo_script.is_file(), str(repo_script))

print("\n=== SUMMARY ===")
passed = sum(1 for _, ok, _ in results if ok)
print(f"{passed}/{len(results)} checks passed")
failed = [n for n, ok, _ in results if not ok]
if failed:
    print("FAILED: " + ", ".join(failed))
    raise SystemExit(1)
print("ALL ADMIN FUNCTIONAL PROBES PASSED")
