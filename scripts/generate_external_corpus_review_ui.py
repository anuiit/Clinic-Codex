#!/usr/bin/env python3
"""Generate a local HTML curation UI for external corpus decisions.

The generated HTML is static and self-contained. It does not mutate source
images or repository files. Browser-side edits are exported as a CSV compatible
with `scripts/validate_external_corpus_decisions.py`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DECISION_HEADERS = [
    "item_type",
    "item_id",
    "current_status",
    "source_dataset",
    "source_folder",
    "path",
    "active_class_name",
    "decision",
    "target_class",
    "notes",
    "reviewer",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def file_url(path: str) -> str:
    if not path:
        return ""
    return Path(path).absolute().as_uri()


def make_thumbnail_url(path: str, thumbnail_dir: Path | None, cache: dict[str, str]) -> str:
    if not path:
        return ""
    if path in cache:
        return cache[path]
    fallback = file_url(path)
    if thumbnail_dir is None:
        cache[path] = fallback
        return fallback
    try:
        from PIL import Image, ImageOps

        source = Path(path)
        if not source.exists() or not source.is_file():
            cache[path] = fallback
            return fallback
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(path.encode("utf-8")).hexdigest()[:24]
        target = thumbnail_dir / f"{digest}.jpg"
        if not target.exists():
            with Image.open(source) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                img.thumbnail((360, 360))
                img.save(target, format="JPEG", quality=88, optimize=True)
        cache[path] = f"{thumbnail_dir.name}/{target.name}"
        return cache[path]
    except Exception:
        cache[path] = fallback
        return fallback


def image_payload(image: dict[str, Any], thumbnail_dir: Path | None = None, thumbnail_cache: dict[str, str] | None = None) -> dict[str, Any]:
    cache = thumbnail_cache if thumbnail_cache is not None else {}
    path = image.get("path", "")
    return {
        "source": image.get("source", ""),
        "source_folder": image.get("class_dir", ""),
        "path": path,
        "url": make_thumbnail_url(path, thumbnail_dir, cache),
        "original_url": file_url(path),
        "active_class_name": image.get("matched_active_class") or "",
        "status": image.get("status", ""),
        "status_reason": image.get("status_reason", ""),
        "sha256_bytes": image.get("sha256_bytes") or "",
        "sha256_pixels": image.get("sha256_pixels") or "",
    }


def build_payload(plan_dir: Path, decisions_path: Path, thumbnail_dir: Path | None = None) -> dict[str, Any]:
    manifest = load_json(plan_dir / "import_manifest.preview.json")
    decisions = read_csv(decisions_path)
    weak_rows = read_csv(plan_dir / "weak_classes.csv") if (plan_dir / "weak_classes.csv").exists() else []
    conflict_rows = read_csv(plan_dir / "duplicate_conflicts.csv") if (plan_dir / "duplicate_conflicts.csv").exists() else []

    images = manifest.get("images", [])
    by_path = {image.get("path", ""): image for image in images}
    thumbnail_cache: dict[str, str] = {}
    active_classes = sorted(
        {
            *(manifest.get("class_counts") or {}).keys(),
            *(image.get("matched_active_class") for image in images if image.get("matched_active_class")),
        }
    )

    conflict_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in conflict_rows:
        key = row.get("sha256_pixels") or row.get("sha256_bytes") or row.get("path", "")
        image = by_path.get(row.get("path", ""), {})
        merged = {**row, **image_payload(image or {"path": row.get("path", "")}, thumbnail_dir, thumbnail_cache)}
        conflict_groups[key].append(merged)

    unmapped_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in images:
        if image.get("status") == "unmapped_extra":
            unmapped_groups[f"{image.get('source', '')}|{image.get('class_dir', '')}"].append(image_payload(image, thumbnail_dir, thumbnail_cache))

    samples_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in images:
        class_name = image.get("matched_active_class")
        if not class_name:
            continue
        if image.get("status") in {"selected", "conflict_cross_class", "suspected_reencode"} and len(samples_by_class[class_name]) < 10:
            samples_by_class[class_name].append(image_payload(image, thumbnail_dir, thumbnail_cache))

    weak_classes = []
    for row in weak_rows:
        class_name = row.get("class_name", "")
        weak_classes.append({**row, "samples": samples_by_class.get(class_name, [])})

    spotcheck_rows = read_csv(plan_dir / "suspected_reencode_spotcheck.csv") if (plan_dir / "suspected_reencode_spotcheck.csv").exists() else []
    spotchecks = [{**row, "url": make_thumbnail_url(row.get("path", ""), thumbnail_dir, thumbnail_cache), "original_url": file_url(row.get("path", ""))} for row in spotcheck_rows]

    return {
        "headers": DECISION_HEADERS,
        "manifestSummary": {
            "image_count": manifest.get("image_count"),
            "status_counts": manifest.get("status_counts", {}),
            "blockers": manifest.get("blockers", []),
            "promotable": manifest.get("promotable"),
        },
        "activeClasses": active_classes,
        "decisions": decisions,
        "conflictGroups": [
            {"group_id": key, "images": sorted(items, key=lambda item: (item.get("active_class_name", ""), item.get("path", "")))}
            for key, items in sorted(conflict_groups.items(), key=lambda item: item[0])
        ],
        "unmappedGroups": [
            {"item_id": key, "images": sorted(items, key=lambda item: item.get("path", ""))}
            for key, items in sorted(unmapped_groups.items(), key=lambda item: item[0])
        ],
        "weakClasses": weak_classes,
        "spotchecks": spotchecks,
        "decisionCounts": dict(Counter(row.get("item_type", "") for row in decisions)),
    }


def html_document(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>External corpus review UI</title>
<style>
:root {{ color-scheme: dark; --bg:#0f1115; --panel:#181b22; --muted:#9ca3af; --text:#e5e7eb; --accent:#60a5fa; --ok:#34d399; --bad:#f87171; --warn:#fbbf24; --line:#2c3440; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background:var(--bg); color:var(--text); }}
header {{ position: sticky; top:0; z-index:10; background:rgba(15,17,21,.96); border-bottom:1px solid var(--line); padding:14px 18px; }}
h1 {{ margin:0 0 8px; font-size:20px; }}
.toolbar {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; }}
button, select, input, textarea {{ background:#111827; color:var(--text); border:1px solid var(--line); border-radius:8px; padding:8px 10px; }}
button {{ cursor:pointer; }}
button.primary {{ background:#1d4ed8; border-color:#2563eb; }}
button.good {{ background:#065f46; border-color:#047857; }}
button.bad {{ background:#7f1d1d; border-color:#991b1b; }}
button.warn {{ background:#78350f; border-color:#92400e; }}
main {{ padding:18px; }}
.grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap:14px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:12px; box-shadow: 0 4px 18px rgba(0,0,0,.25); }}
.card.done {{ border-color:rgba(52,211,153,.7); }}
.card.missing {{ border-color:rgba(248,113,113,.7); }}
.meta {{ font-size:12px; color:var(--muted); overflow-wrap:anywhere; }}
.badge {{ display:inline-block; padding:2px 7px; border:1px solid var(--line); border-radius:999px; font-size:12px; color:var(--muted); margin:2px; }}
.badge.ok {{ color:var(--ok); border-color:var(--ok); }} .badge.bad {{ color:var(--bad); border-color:var(--bad); }} .badge.warn {{ color:var(--warn); border-color:var(--warn); }}
img.thumb {{ max-width:100%; max-height:220px; object-fit:contain; background:#050607; border:1px solid var(--line); border-radius:10px; display:block; margin:8px auto; }}
.row {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:8px 0; }}
.row > * {{ flex:1; min-width:120px; }}
textarea {{ width:100%; min-height:54px; }}
.hidden {{ display:none !important; }}
.progress {{ height:10px; background:#111827; border-radius:999px; overflow:hidden; width:240px; border:1px solid var(--line); }}
.progress > div {{ height:100%; background:linear-gradient(90deg, var(--bad), var(--warn), var(--ok)); width:0%; }}
.section-title {{ margin:16px 0 10px; display:flex; align-items:center; gap:10px; }}
.split {{ display:grid; grid-template-columns: 1fr 1fr; gap:10px; }}
@media (max-width: 800px) {{ .split {{ grid-template-columns:1fr; }} }}
a {{ color:var(--accent); }}
.code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; background:#0b0d12; padding:2px 5px; border-radius:5px; }}
.notice {{ border-left:4px solid var(--warn); background:#1b1608; padding:10px 12px; margin:12px 0; border-radius:8px; }}
</style>
</head>
<body>
<header>
  <h1>External corpus review UI</h1>
  <div class="toolbar">
    <button data-tab="conflicts" class="primary">Conflits</button>
    <button data-tab="unmapped">Unmapped</button>
    <button data-tab="weak">Weak classes</button>
    <button data-tab="spotcheck">Re-encode spot-check</button>
    <button data-tab="all">Toutes décisions</button>
    <span class="badge" id="counts"></span>
    <div class="progress"><div id="bar"></div></div>
    <input id="reviewer" placeholder="reviewer" style="max-width:160px" />
    <button class="good" id="exportCsv">Exporter CSV</button>
    <button id="copyCsv">Copier CSV</button>
    <button class="bad" id="resetLocal">Reset local</button>
  </div>
</header>
<main>
  <div class="notice">
    Cette page ne modifie aucun fichier automatiquement. Après tes choix, clique <b>Exporter CSV</b> puis remplace <span class="code">reviewer_decisions.csv</span> par le fichier téléchargé.
  </div>
  <section id="summary"></section>
  <section id="conflicts" class="tab"></section>
  <section id="unmapped" class="tab hidden"></section>
  <section id="weak" class="tab hidden"></section>
  <section id="spotcheck" class="tab hidden"></section>
  <section id="all" class="tab hidden"></section>
</main>
<datalist id="classes"></datalist>
<script id="payload" type="application/json">{data}</script>
<script>
const payload = JSON.parse(document.getElementById('payload').textContent);
const STORAGE_KEY = 'external-corpus-review-ui-v1:' + location.pathname;
const byKey = new Map(payload.decisions.map(r => [r.item_type + '|' + r.item_id, {{...r}}]));
const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}');
for (const [k, v] of Object.entries(saved.decisions || {{}})) if (byKey.has(k)) Object.assign(byKey.get(k), v);
if (saved.reviewer) document.getElementById('reviewer').value = saved.reviewer;
document.getElementById('classes').innerHTML = payload.activeClasses.map(c => `<option value="${{esc(c)}}"></option>`).join('');

document.querySelectorAll('[data-tab]').forEach(btn => btn.addEventListener('click', () => showTab(btn.dataset.tab)));
document.getElementById('exportCsv').addEventListener('click', exportCsv);
document.getElementById('copyCsv').addEventListener('click', async () => {{ await navigator.clipboard.writeText(toCsv()); alert('CSV copié'); }});
document.getElementById('resetLocal').addEventListener('click', () => {{ if(confirm('Effacer les choix sauvegardés dans ce navigateur ?')) {{ localStorage.removeItem(STORAGE_KEY); location.reload(); }} }});
document.getElementById('reviewer').addEventListener('input', () => {{ for (const row of byKey.values()) if (!row.reviewer) row.reviewer = document.getElementById('reviewer').value; save(); render(); }});

function esc(s) {{ return String(s ?? '').replace(/[&<>"']/g, m => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[m])); }}
function key(type, id) {{ return type + '|' + id; }}
function row(type, id) {{ return byKey.get(key(type, id)); }}
function isDone(r) {{
  if (!r || !r.decision) return false;
  if (['keep_as','map_to_existing','create_new_class'].includes(r.decision)) return !!r.target_class;
  return true;
}}
function save() {{
  const decisions = {{}};
  for (const [k, r] of byKey) decisions[k] = {{decision:r.decision, target_class:r.target_class, notes:r.notes, reviewer:r.reviewer}};
  localStorage.setItem(STORAGE_KEY, JSON.stringify({{reviewer:document.getElementById('reviewer').value, decisions}}));
  updateProgress();
}}
function updateRow(type, id, patch) {{
  const r = row(type, id); if (!r) return;
  Object.assign(r, patch);
  if (!r.reviewer) r.reviewer = document.getElementById('reviewer').value;
  save();
  const card = document.querySelector(`[data-decision-key="${{CSS.escape(key(type,id))}}"]`);
  if (card) card.className = 'card ' + (isDone(r) ? 'done' : 'missing');
}}
function updateProgress() {{
  const rows = [...byKey.values()];
  const done = rows.filter(isDone).length;
  const pct = rows.length ? Math.round(done * 100 / rows.length) : 100;
  document.getElementById('counts').textContent = `${{done}}/${{rows.length}} décisions (${{pct}}%)`;
  document.getElementById('bar').style.width = pct + '%';
}}
function showTab(id) {{ document.querySelectorAll('.tab').forEach(t => t.classList.toggle('hidden', t.id !== id)); }}
function imageCard(img, extra='') {{
  return `<div class="card"><img class="thumb" src="${{esc(img.url)}}" onerror="this.insertAdjacentHTML('afterend','<p class=&quot;meta&quot;>image non chargée</p>'); this.style.display='none'" />
    <div><span class="badge">${{esc(img.active_class_name || 'unmapped')}}</span><span class="badge">${{esc(img.status)}}</span></div>
    <div class="meta">${{esc(img.source)}} / ${{esc(img.source_folder)}}</div>
    <div class="meta">${{esc(img.path)}}</div>${{extra}}</div>`;
}}
function decisionControls(r, options) {{
  const optionHtml = [''].concat(options).map(o => `<option value="${{esc(o)}}" ${{r.decision===o?'selected':''}}>${{esc(o || 'à décider')}}</option>`).join('');
  return `<div class="row">
    <select onchange="updateRow('${{esc(r.item_type)}}','${{esc(r.item_id)}}',{{decision:this.value}})">${{optionHtml}}</select>
    <input list="classes" placeholder="target_class" value="${{esc(r.target_class)}}" onchange="updateRow('${{esc(r.item_type)}}','${{esc(r.item_id)}}',{{target_class:this.value}})" />
  </div>
  <textarea placeholder="notes" onchange="updateRow('${{esc(r.item_type)}}','${{esc(r.item_id)}}',{{notes:this.value}})">${{esc(r.notes)}}</textarea>`;
}}
function renderSummary() {{
 const s = payload.manifestSummary;
 document.getElementById('summary').innerHTML = `<div class="card"><b>Résumé</b> — images: ${{s.image_count}}, promotable: ${{s.promotable}}<br>
   Statuts: ${{Object.entries(s.status_counts).map(([k,v]) => `<span class="badge">${{esc(k)}}: ${{v}}</span>`).join('')}}<br>
   Blockers: ${{s.blockers.map(b => `<span class="badge bad">${{esc(b)}}</span>`).join('')}}</div>`;
}}
function renderConflicts() {{
 const html = [`<div class="section-title"><h2>Conflits image</h2><span class="badge bad">${{payload.conflictGroups.length}} groupes</span></div>`];
 for (const g of payload.conflictGroups) {{
   html.push(`<div class="card"><div class="meta">hash: ${{esc(g.group_id)}}</div><div class="grid">`);
   for (const img of g.images) {{
     const r = row('image_conflict', img.path); if (!r) continue;
     html.push(`<div class="card ${{isDone(r)?'done':'missing'}}" data-decision-key="${{esc(key('image_conflict', img.path))}}">
       <img class="thumb" src="${{esc(img.url)}}" />
       <span class="badge bad">${{esc(img.status)}}</span><span class="badge">${{esc(img.active_class_name)}}</span>
       <div class="meta">${{esc(img.source_folder)}} — ${{esc(img.path)}}</div>
       ${{decisionControls(r, ['keep_as','quarantine'])}}
     </div>`);
   }}
   html.push(`</div></div>`);
 }}
 document.getElementById('conflicts').innerHTML = html.join('');
}}
function renderUnmapped() {{
 const html = [`<div class="section-title"><h2>Classes non mappées</h2><span class="badge warn">${{payload.unmappedGroups.length}}</span></div>`];
 for (const g of payload.unmappedGroups) {{
   const r = row('unmapped_class', g.item_id); if (!r) continue;
   html.push(`<div class="card ${{isDone(r)?'done':'missing'}}" data-decision-key="${{esc(key('unmapped_class', g.item_id))}}">
     <h3>${{esc(g.item_id)}}</h3>${{decisionControls(r, ['map_to_existing','create_new_class','quarantine'])}}
     <div class="grid">${{g.images.slice(0,12).map(img => imageCard(img)).join('')}}</div>
   </div>`);
 }}
 document.getElementById('unmapped').innerHTML = html.join('');
}}
function renderWeak() {{
 const html = [`<div class="section-title"><h2>Classes faibles</h2><span class="badge warn">${{payload.weakClasses.length}}</span><button onclick="bulkWeak('accept_weak')">Tout accepter weak</button><button onclick="bulkWeak('exclude_until_more_data')">Tout exclure weak</button></div>`];
 for (const wc of payload.weakClasses) {{
   const r = row('weak_class', wc.class_name); if (!r) continue;
   html.push(`<div class="card ${{isDone(r)?'done':'missing'}}" data-decision-key="${{esc(key('weak_class', wc.class_name))}}">
    <h3>${{esc(wc.class_name)}}</h3><div class="meta">raw=${{esc(wc.raw_count)}} unique=${{esc(wc.unique_count)}} train=${{esc(wc.train_count)}} val=${{esc(wc.val_count)}} reasons=${{esc(wc.reasons)}}</div>
    ${{decisionControls(r, ['accept_weak','exclude_until_more_data'])}}
    <div class="grid">${{(wc.samples||[]).slice(0,8).map(img => imageCard(img)).join('')}}</div>
   </div>`);
 }}
 document.getElementById('weak').innerHTML = html.join('');
}}
function bulkWeak(decision) {{ for (const wc of payload.weakClasses) updateRow('weak_class', wc.class_name, {{decision, target_class:''}}); renderWeak(); }}
function renderSpotcheck() {{
 document.getElementById('spotcheck').innerHTML = `<div class="section-title"><h2>Spot-check re-encode</h2><span class="badge">${{payload.spotchecks.length}}</span></div><div class="grid">${{payload.spotchecks.map(s => imageCard({{...s, active_class_name:s.class_name, status:'suspected_reencode', source:s.source_dataset, source_folder:s.source_folder, url:s.url, path:s.path}}, `<div class="meta">${{esc(s.status_reason)}}</div>`)).join('')}}</div>`;
}}
function renderAll() {{
 const rows = [...byKey.values()];
 document.getElementById('all').innerHTML = `<div class="section-title"><h2>Toutes décisions</h2></div><div class="grid">${{rows.map(r => `<div class="card ${{isDone(r)?'done':'missing'}}" data-decision-key="${{esc(key(r.item_type,r.item_id))}}"><b>${{esc(r.item_type)}}</b><div class="meta">${{esc(r.item_id)}}</div>${{decisionControls(r, r.item_type==='image_conflict'?['keep_as','quarantine']:r.item_type==='unmapped_class'?['map_to_existing','create_new_class','quarantine']:r.item_type==='weak_class'?['accept_weak','exclude_until_more_data']:['quarantine'])}}</div>`).join('')}}</div>`;
}}
function render() {{ renderSummary(); renderConflicts(); renderUnmapped(); renderWeak(); renderSpotcheck(); renderAll(); updateProgress(); }}
function csvEscape(v) {{ const s = String(v ?? ''); return /[",\\n\\r]/.test(s) ? '"' + s.replaceAll('"','""') + '"' : s; }}
function toCsv() {{
  const rows = [...byKey.values()].map(r => payload.headers.map(h => csvEscape(r[h] || '')).join(','));
  return payload.headers.join(',') + '\\n' + rows.join('\\n') + '\\n';
}}
function exportCsv() {{
  const blob = new Blob([toCsv()], {{type:'text/csv;charset=utf-8'}});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'reviewer_decisions.csv'; a.click(); URL.revokeObjectURL(a.href);
}}
render();
</script>
</body>
</html>"""


def generate(plan_dir: Path, decisions_path: Path, output_html: Path) -> dict[str, Any]:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    thumbnail_dir = output_html.parent / "review_thumbnails"
    payload = build_payload(plan_dir, decisions_path, thumbnail_dir)
    output_html.write_text(html_document(payload), encoding="utf-8")
    thumbnail_count = len(list(thumbnail_dir.glob("*.jpg"))) if thumbnail_dir.exists() else 0
    return {
        "output_html": str(output_html),
        "thumbnail_dir": str(thumbnail_dir),
        "thumbnail_count": thumbnail_count,
        "decision_rows": len(payload["decisions"]),
        "conflict_groups": len(payload["conflictGroups"]),
        "unmapped_groups": len(payload["unmappedGroups"]),
        "weak_classes": len(payload["weakClasses"]),
        "spotchecks": len(payload["spotchecks"]),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--decisions", help="CSV decisions file; defaults to <plan-dir>/reviewer_decisions.csv")
    parser.add_argument("--output-html", help="Output HTML; defaults to <plan-dir>/review_ui.html")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan_dir = Path(args.plan_dir).expanduser()
    decisions = Path(args.decisions).expanduser() if args.decisions else plan_dir / "reviewer_decisions.csv"
    output = Path(args.output_html).expanduser() if args.output_html else plan_dir / "review_ui.html"
    try:
        summary = generate(plan_dir, decisions, output)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
