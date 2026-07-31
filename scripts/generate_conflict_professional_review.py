#!/usr/bin/env python3
"""Generate a non-technical professional review package for duplicate conflicts.

The package is self-contained enough to send as a folder/zip:
- index.html: large, simple, high-contrast review UI
- images/: local enlarged JPEG copies, so no /mnt/f paths are needed
- README.txt: short instructions

It does not modify source images or make import decisions. It only collects
human answers that can later be translated into reviewer decisions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_name(value: str, max_len: int = 80) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    text = "".join(keep).strip("_") or "item"
    return text[:max_len]


def make_review_image(source_path: str, output_dir: Path) -> str:
    source = Path(source_path)
    digest = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:16]
    stem = safe_name(source.stem, 48)
    target = output_dir / f"{digest}_{stem}.jpg"
    if target.exists():
        return f"images/{target.name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageOps

        with Image.open(source) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            # Enlarged but not blurry beyond reason: nearest preserves pictogram edges.
            max_side = max(img.size)
            if max_side < 900:
                scale = max(1, 900 // max_side)
                img = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
            img.thumbnail((1200, 1200), Image.Resampling.NEAREST)
            img.save(target, format="JPEG", quality=92)
    except Exception:
        # Fallback keeps the package usable even if Pillow cannot decode one file.
        shutil.copy2(source, target)
    return f"images/{target.name}"


def build_groups(plan_dir: Path, package_dir: Path) -> list[dict[str, Any]]:
    manifest = load_json(plan_dir / "import_manifest.preview.json")
    conflict_rows = read_csv(plan_dir / "duplicate_conflicts.csv")
    by_path = {image.get("path", ""): image for image in manifest.get("images", [])}
    image_dir = package_dir / "images"

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in conflict_rows:
        if row.get("status") != "conflict_cross_class":
            continue
        key = row.get("sha256_pixels") or row.get("sha256_bytes") or row.get("path", "")
        image = by_path.get(row.get("path", ""), {})
        path = row.get("path", "")
        groups[key].append(
            {
                "path": path,
                "image": make_review_image(path, image_dir),
                "current_label": row.get("active_class_name") or image.get("matched_active_class") or "",
                "folder": row.get("source_folder") or image.get("class_dir") or "",
                "source": row.get("source_dataset") or image.get("source") or "",
            }
        )

    output = []
    for idx, (group_id, items) in enumerate(sorted(groups.items(), key=lambda pair: pair[0]), start=1):
        labels = sorted({item["current_label"] for item in items if item["current_label"]})
        output.append(
            {
                "group_number": idx,
                "group_id": group_id,
                "labels": labels,
                "items": sorted(items, key=lambda item: (item["current_label"], item["folder"], item["path"])),
            }
        )
    return output


def html_doc(groups: list[dict[str, Any]]) -> str:
    payload = json.dumps(groups, ensure_ascii=False).replace("</", "<\\/")
    template = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Review professionnelle — images ambiguës</title>
<style>
:root { --bg:#f7f7f2; --panel:#ffffff; --text:#111827; --muted:#4b5563; --line:#cbd5e1; --blue:#1d4ed8; --green:#166534; --red:#991b1b; --orange:#9a3412; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font-family:Arial, Helvetica, sans-serif; font-size:22px; line-height:1.45; }
header { position:sticky; top:0; z-index:10; background:#fff; border-bottom:3px solid var(--line); padding:18px 24px; }
h1 { margin:0 0 10px; font-size:34px; }
.toolbar { display:flex; flex-wrap:wrap; gap:12px; align-items:center; }
button, select, input, textarea { font:inherit; border:2px solid var(--line); border-radius:12px; padding:12px 14px; background:#fff; color:var(--text); }
button { cursor:pointer; font-weight:700; }
button.primary { background:var(--blue); color:#fff; border-color:var(--blue); }
button.good { background:var(--green); color:#fff; border-color:var(--green); }
main { max-width:1600px; margin:0 auto; padding:22px; }
.card { background:var(--panel); border:3px solid var(--line); border-radius:18px; padding:18px; margin:20px 0; box-shadow:0 4px 16px rgba(0,0,0,.08); }
.card.done { border-color:var(--green); }
.badge { display:inline-block; border:2px solid var(--line); border-radius:999px; padding:5px 12px; margin:4px; font-weight:700; background:#f8fafc; }
.badge.warn { border-color:var(--orange); color:var(--orange); }
.images { display:grid; grid-template-columns:repeat(auto-fit, minmax(520px, 1fr)); gap:20px; align-items:start; }
.figure { background:#fff; border:2px solid var(--line); border-radius:16px; padding:14px; }
.figure img { width:100%; max-height:700px; object-fit:contain; background:#fff; border:3px solid #111827; border-radius:12px; }
.label { font-size:30px; font-weight:800; margin:8px 0; color:var(--blue); }
.small { font-size:16px; color:var(--muted); overflow-wrap:anywhere; }
.decision { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:16px; }
.decision > * { width:100%; }
textarea { min-height:110px; grid-column:1 / -1; }
.progress { font-weight:800; }
.instructions { background:#fff7ed; border-left:8px solid var(--orange); padding:16px 18px; border-radius:12px; margin:16px 0; }
@media (max-width:900px) { .images, .decision { grid-template-columns:1fr; } body { font-size:19px; } h1 { font-size:28px; } }
@media print { header { position:static; } .card { break-inside:avoid; } }
</style>
</head>
<body>
<header>
  <h1>Review professionnelle — images ambiguës</h1>
  <div class="toolbar">
    <span class="progress" id="progress"></span>
    <input id="reviewer" placeholder="Nom du reviewer" />
    <button class="primary" id="prev">← Précédent</button>
    <button class="primary" id="next">Suivant →</button>
    <button class="good" id="export">Exporter les réponses CSV</button>
    <button id="showAll">Tout afficher</button>
  </div>
</header>
<main>
  <div class="instructions">
    <b>But :</b> ces images sont identiques ou quasi identiques, mais elles apparaissent sous plusieurs noms/classes. Merci d'indiquer si l'image correspond à une seule classe, plusieurs classes, une erreur, ou si vous n'êtes pas sûr.<br />
    Les images sont volontairement grandes. Aucune connaissance technique n'est nécessaire.
  </div>
  <div id="content"></div>
</main>
<script id="groups" type="application/json">__PAYLOAD__</script>
<script>
const groups = JSON.parse(document.getElementById('groups').textContent);
const STORAGE = 'professional-conflict-review-v1:' + location.pathname;
let state = JSON.parse(localStorage.getItem(STORAGE) || '{"answers":{},"index":0,"showAll":false,"reviewer":""}');
const decisions = ['', 'Une seule classe correcte', 'Plusieurs classes dans l’image', 'Erreur de classement / à exclure', 'Je ne suis pas sûr'];
function esc(s) { return String(s ?? '').replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])); }
function save() { localStorage.setItem(STORAGE, JSON.stringify(state)); renderProgress(); }
function answer(g) { return state.answers[g.group_number] || {decision:'', chosen_label:'', notes:'', reviewer:''}; }
function setAnswer(n, patch) { state.answers[n] = {...answer({group_number:n}), ...patch}; if (!state.answers[n].reviewer) state.answers[n].reviewer = document.getElementById('reviewer').value; save(); render(); }
function renderProgress() { const done = groups.filter(g => answer(g).decision).length; document.getElementById('progress').textContent = `${done} / ${groups.length} groupes répondus`; }
function groupHtml(g) {
 const a = answer(g);
 const labelOptions = [''].concat(g.labels).map(l => `<option value="${esc(l)}" ${a.chosen_label===l?'selected':''}>${esc(l || 'choisir une classe si utile')}</option>`).join('');
 const decisionOptions = decisions.map(d => `<option value="${esc(d)}" ${a.decision===d?'selected':''}>${esc(d || 'choisir une réponse')}</option>`).join('');
 return `<section class="card ${a.decision?'done':''}">
   <h2>Groupe ${g.group_number} / ${groups.length}</h2>
   <div>${g.labels.map(l => `<span class="badge warn">${esc(l)}</span>`).join('')}</div>
   <div class="images">${g.items.map(item => `<div class="figure"><img src="${esc(item.image)}" alt="image à comparer" /><div class="label">Classe actuelle : ${esc(item.current_label)}</div><div class="small">Dossier : ${esc(item.folder)}</div></div>`).join('')}</div>
   <div class="decision">
     <select onchange="setAnswer(${g.group_number}, {decision:this.value})">${decisionOptions}</select>
     <select onchange="setAnswer(${g.group_number}, {chosen_label:this.value})">${labelOptions}</select>
     <textarea placeholder="Commentaire libre" onchange="setAnswer(${g.group_number}, {notes:this.value})">${esc(a.notes)}</textarea>
   </div>
 </section>`;
}
function render() {
 document.getElementById('reviewer').value = state.reviewer || '';
 const content = document.getElementById('content');
 if (state.showAll) content.innerHTML = groups.map(groupHtml).join('');
 else content.innerHTML = groupHtml(groups[Math.max(0, Math.min(state.index, groups.length-1))]);
 renderProgress();
}
function csvEscape(v) { const s = String(v ?? ''); return /[",\\n\\r]/.test(s) ? '"' + s.replaceAll('"','""') + '"' : s; }
function toCsv() {
 const headers = ['group_number','labels','decision','chosen_label','notes','reviewer','image_paths'];
 const rows = groups.map(g => { const a = answer(g); return [g.group_number, g.labels.join('|'), a.decision, a.chosen_label, a.notes, a.reviewer || state.reviewer || '', g.items.map(i => i.path).join('|')].map(csvEscape).join(','); });
 return headers.join(',') + '\\n' + rows.join('\\n') + '\\n';
}
document.getElementById('reviewer').addEventListener('input', e => { state.reviewer = e.target.value; save(); });
document.getElementById('prev').onclick = () => { state.index = Math.max(0, (state.index||0)-1); state.showAll=false; save(); render(); };
document.getElementById('next').onclick = () => { state.index = Math.min(groups.length-1, (state.index||0)+1); state.showAll=false; save(); render(); };
document.getElementById('showAll').onclick = () => { state.showAll = !state.showAll; save(); render(); };
document.getElementById('export').onclick = () => { const blob = new Blob([toCsv()], {type:'text/csv;charset=utf-8'}); const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download='professional_conflict_answers.csv'; a.click(); URL.revokeObjectURL(a.href); };
render();
</script>
</body>
</html>"""
    return template.replace("__PAYLOAD__", payload)

def readme_text(groups: list[dict[str, Any]]) -> str:
    return f"""Review professionnelle — images ambiguës

Ouvrir le fichier index.html dans un navigateur.

Objectif : répondre aux {len(groups)} groupes d'images ambiguës.
Les images sont volontairement agrandies et stockées localement dans le dossier images/.

Pour chaque groupe, choisir :
- Une seule classe correcte
- Plusieurs classes dans l'image
- Erreur de classement / à exclure
- Je ne suis pas sûr

À la fin, cliquer sur "Exporter les réponses CSV" et renvoyer le fichier professional_conflict_answers.csv.

Ne pas renommer les fichiers du dossier.
"""


def generate(plan_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = build_groups(plan_dir, output_dir)
    (output_dir / "index.html").write_text(html_doc(groups), encoding="utf-8")
    (output_dir / "README.txt").write_text(readme_text(groups), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "index_html": str(output_dir / "index.html"),
        "readme": str(output_dir / "README.txt"),
        "image_dir": str(output_dir / "images"),
        "group_count": len(groups),
        "image_count": sum(len(group["items"]) for group in groups),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = generate(Path(args.plan_dir).expanduser(), Path(args.output_dir).expanduser())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
