#!/usr/bin/env python3
"""Generate a portable visual comparison report from model prediction rows."""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
from pathlib import Path
from typing import Any, Iterable


MODEL_ORDER = ("historical", "v5", "augmented")
MODEL_LABELS = {
    "historical": "Historique",
    "v5": "V5",
    "augmented": "Augmenté",
}


def _prediction(row: dict[str, Any], model: str) -> dict[str, Any] | None:
    value = row.get(model)
    if value is None and isinstance(row.get("models"), dict):
        value = row["models"].get(model)
    return value if isinstance(value, dict) else None


def load_prediction_rows(input_path: Path) -> list[dict[str, Any]]:
    """Load either a JSON list or an object containing a ``rows`` list."""
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("prediction JSON must be a list or contain a 'rows' list")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("every prediction row must be a JSON object")
    return rows


def _image_data_uri(raw_path: Any, source_dir: Path) -> tuple[str, str]:
    if not isinstance(raw_path, str) or not raw_path:
        return "", "Chemin d’image absent"
    image_path = Path(raw_path)
    if not image_path.is_absolute():
        image_path = source_dir / image_path
    try:
        image_bytes = image_path.read_bytes()
    except OSError:
        return "", f"Image introuvable : {raw_path}"
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}", ""


def _normalise_rows(
    rows: Iterable[dict[str, Any]], source_dir: Path, max_images: int
) -> tuple[list[dict[str, Any]], list[str]]:
    normalised: list[dict[str, Any]] = []
    available_models: set[str] = set()
    for row in list(rows)[:max_images]:
        truth = str(row.get("truth", ""))
        models: dict[str, Any] = {}
        for model in MODEL_ORDER:
            prediction = _prediction(row, model)
            if prediction is None:
                models[model] = None
                continue
            available_models.add(model)
            pred = str(prediction.get("pred", ""))
            top3_value = prediction.get("top3", [])
            top3 = [str(value) for value in top3_value] if isinstance(top3_value, list) else []
            models[model] = {
                "pred": pred,
                "top3": top3,
                "correct": bool(prediction.get("correct", pred == truth)),
            }
        data_uri, image_error = _image_data_uri(row.get("path"), source_dir)
        normalised.append(
            {
                "path": str(row.get("path", "")),
                "truth": truth,
                "image": data_uri,
                "imageError": image_error,
                "models": models,
            }
        )
    ordered_models = [model for model in MODEL_ORDER if model in available_models]
    return normalised, ordered_models


def _metrics(rows: list[dict[str, Any]], models: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for model in models:
        predictions = [row["models"][model] for row in rows if row["models"][model] is not None]
        total = len(predictions)
        correct = sum(bool(prediction["correct"]) for prediction in predictions)
        top3 = sum(
            row["truth"] in row["models"][model]["top3"]
            for row in rows
            if row["models"][model] is not None
        )
        result[model] = {
            "total": total,
            "correct": correct,
            "top1": correct / total if total else 0.0,
            "top3": top3 / total if total else 0.0,
        }
    return result


def _json_for_script(value: Any) -> str:
    # Prevent row content from prematurely closing the script element.
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def generate_html(
    rows: Iterable[dict[str, Any]],
    *,
    source_dir: Path,
    max_images: int = 200,
) -> str:
    if max_images < 1:
        raise ValueError("max_images must be at least 1")
    report_rows, available_models = _normalise_rows(rows, source_dir, max_images)
    metrics = _metrics(report_rows, available_models)
    available_model_labels = {model: MODEL_LABELS[model] for model in available_models}
    classes = sorted({row["truth"] for row in report_rows if row["truth"]})

    metric_cards = "".join(
        (
            '<article class="metric">'
            f"<strong>{html.escape(MODEL_LABELS[model])}</strong>"
            f"<span>Top-1&nbsp;: {metrics[model]['top1']:.1%}</span>"
            f"<span>Top-3&nbsp;: {metrics[model]['top3']:.1%}</span>"
            f"<small>{metrics[model]['correct']} / {metrics[model]['total']} correctes</small>"
            "</article>"
        )
        for model in available_models
    )
    model_options = "".join(
        f'<option value="{model}">{html.escape(MODEL_LABELS[model])}</option>'
        for model in available_models
    )
    class_options = "".join(
        f'<option value="{html.escape(value, quote=True)}">{html.escape(value)}</option>'
        for value in classes
    )

    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Comparaison des modèles</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, system-ui, sans-serif; color: #172033; background: #f3f5f8; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; }}
    header, main {{ width: min(1180px, calc(100% - 32px)); margin: auto; }}
    header {{ padding: 28px 0 18px; }}
    h1 {{ margin: 0 0 6px; font-size: clamp(1.5rem, 3vw, 2.2rem); }}
    .muted, small {{ color: #657087; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-top: 20px; }}
    .metric, .toolbar, .viewer {{ background: white; border: 1px solid #dfe4ec; border-radius: 12px; box-shadow: 0 3px 16px #1520380a; }}
    .metric {{ display: grid; gap: 4px; padding: 14px; }}
    .metric strong {{ margin-bottom: 4px; }}
    .toolbar {{ position: sticky; top: 8px; z-index: 2; display: grid; grid-template-columns: 1.4fr repeat(3, minmax(130px, .7fr)); gap: 10px; padding: 12px; }}
    label {{ display: grid; gap: 5px; font-size: .76rem; font-weight: 700; color: #4c5870; }}
    input, select, button {{ min-height: 40px; border: 1px solid #cbd2de; border-radius: 8px; background: white; padding: 0 11px; color: inherit; font: inherit; }}
    button {{ cursor: pointer; font-weight: 700; }}
    button:hover:not(:disabled) {{ border-color: #6d7fa6; background: #f5f7fb; }}
    button:disabled {{ cursor: default; opacity: .4; }}
    .viewer {{ margin: 14px 0 32px; overflow: hidden; }}
    .viewer-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 13px 16px; border-bottom: 1px solid #e5e9f0; }}
    .nav {{ display: flex; align-items: center; gap: 8px; }}
    #counter {{ min-width: 90px; text-align: center; font-variant-numeric: tabular-nums; }}
    .content {{ display: grid; grid-template-columns: minmax(260px, 1fr) 1.4fr; min-height: 420px; }}
    .image-pane {{ display: grid; place-items: center; padding: 20px; background: #eef1f6; }}
    .image-pane img {{ display: block; max-width: 100%; max-height: 62vh; object-fit: contain; border-radius: 6px; box-shadow: 0 4px 18px #17203322; }}
    .empty-image {{ text-align: center; color: #7b3541; }}
    .details {{ padding: 20px; overflow: hidden; }}
    .path {{ overflow-wrap: anywhere; font-size: .8rem; color: #657087; }}
    .truth {{ display: inline-flex; gap: 7px; align-items: center; margin: 8px 0 18px; padding: 8px 11px; border-radius: 8px; background: #edf2ff; }}
    .predictions {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(155px, 1fr)); gap: 10px; }}
    .prediction {{ min-width: 0; border: 1px solid; border-radius: 10px; padding: 12px; }}
    .prediction.correct {{ color: #14532d; background: #effaf3; border-color: #8ed2a4; }}
    .prediction.wrong {{ color: #7f1d1d; background: #fff3f3; border-color: #efaaaa; }}
    .prediction.missing {{ color: #657087; background: #f5f6f8; border-color: #d8dde5; }}
    .prediction h3 {{ margin: 0 0 10px; font-size: .9rem; }}
    .prediction .answer {{ display: block; font-size: 1.06rem; font-weight: 800; overflow-wrap: anywhere; }}
    .top3 {{ margin: 12px 0 0; padding-left: 20px; font-size: .83rem; overflow-wrap: anywhere; }}
    .no-results {{ padding: 60px 20px; text-align: center; color: #657087; }}
    kbd {{ border: 1px solid #cbd2de; border-bottom-width: 2px; border-radius: 4px; padding: 1px 5px; background: white; }}
    @media (max-width: 760px) {{
      .toolbar, .content {{ grid-template-columns: 1fr; }}
      .toolbar {{ position: static; }}
      .content {{ min-height: 0; }}
      .image-pane img {{ max-height: 45vh; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Comparaison visuelle des modèles</h1>
    <div class="muted">{len(report_rows)} images intégrées dans ce rapport autonome.</div>
    <section class="metrics" aria-label="Métriques globales">{metric_cards or '<span class="muted">Aucune prédiction disponible.</span>'}</section>
  </header>
  <main>
    <section class="toolbar" aria-label="Filtres">
      <label>Recherche
        <input id="search" type="search" placeholder="Chemin, vérité ou prédiction…">
      </label>
      <label>Modèle
        <select id="modelFilter"><option value="">Tous</option>{model_options}</select>
      </label>
      <label>Résultat
        <select id="resultFilter">
          <option value="">Tous</option>
          <option value="correct">Correct</option>
          <option value="incorrect">Incorrect</option>
        </select>
      </label>
      <label>Classe réelle
        <select id="classFilter"><option value="">Toutes</option>{class_options}</select>
      </label>
    </section>
    <section class="viewer" aria-live="polite">
      <div class="viewer-head">
        <span class="muted">Navigation&nbsp;: <kbd>←</kbd> <kbd>→</kbd></span>
        <div class="nav">
          <button id="previous" type="button" aria-label="Image précédente">←</button>
          <strong id="counter">0 / 0</strong>
          <button id="next" type="button" aria-label="Image suivante">→</button>
        </div>
      </div>
      <div id="card"></div>
    </section>
  </main>
  <script>
    "use strict";
    const rows = {_json_for_script(report_rows)};
    const modelOrder = {_json_for_script(available_models)};
    const modelLabels = {_json_for_script(available_model_labels)};
    const filters = {{
      search: document.getElementById("search"),
      model: document.getElementById("modelFilter"),
      result: document.getElementById("resultFilter"),
      className: document.getElementById("classFilter")
    }};
    const card = document.getElementById("card");
    const counter = document.getElementById("counter");
    const previous = document.getElementById("previous");
    const next = document.getElementById("next");
    let filtered = rows.slice();
    let index = 0;

    function escapeHtml(value) {{
      return String(value).replace(/[&<>"']/g, character => ({{
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }})[character]);
    }}

    function matches(row) {{
      const model = filters.model.value;
      const result = filters.result.value;
      const selectedPrediction = model ? row.models[model] : null;
      if (filters.className.value && row.truth !== filters.className.value) return false;
      if (model && !selectedPrediction) return false;
      if (result) {{
        const expected = result === "correct";
        if (model) {{
          if (selectedPrediction.correct !== expected) return false;
        }} else {{
          const present = modelOrder.map(name => row.models[name]).filter(Boolean);
          if (!present.some(prediction => prediction.correct === expected)) return false;
        }}
      }}
      const query = filters.search.value.trim().toLocaleLowerCase("fr");
      if (!query) return true;
      const text = [row.path, row.truth, ...modelOrder.flatMap(name => {{
        const prediction = row.models[name];
        return prediction ? [prediction.pred, ...prediction.top3] : [];
      }})].join(" ").toLocaleLowerCase("fr");
      return text.includes(query);
    }}

    function predictionHtml(model, prediction) {{
      if (!prediction) return `<article class="prediction missing"><h3>${{escapeHtml(modelLabels[model])}}</h3><span>Non disponible</span></article>`;
      const status = prediction.correct ? "correct" : "wrong";
      const top3 = prediction.top3.length
        ? `<ol class="top3">${{prediction.top3.map(value => `<li>${{escapeHtml(value)}}</li>`).join("")}}</ol>`
        : '<div class="top3">Top-3 non disponible</div>';
      return `<article class="prediction ${{status}}">
        <h3>${{escapeHtml(modelLabels[model])}} · ${{prediction.correct ? "Correct" : "Incorrect"}}</h3>
        <span class="answer">${{escapeHtml(prediction.pred || "—")}}</span>${{top3}}
      </article>`;
    }}

    function render() {{
      const total = filtered.length;
      if (!total) {{
        counter.textContent = "0 / 0";
        previous.disabled = next.disabled = true;
        card.innerHTML = '<div class="no-results">Aucune image ne correspond aux filtres.</div>';
        return;
      }}
      index = Math.max(0, Math.min(index, total - 1));
      const row = filtered[index];
      counter.textContent = `${{index + 1}} / ${{total}}`;
      previous.disabled = index === 0;
      next.disabled = index === total - 1;
      const image = row.image
        ? `<img src="${{row.image}}" alt="Image à classifier">`
        : `<div class="empty-image">${{escapeHtml(row.imageError)}}</div>`;
      card.innerHTML = `<div class="content">
        <div class="image-pane">${{image}}</div>
        <div class="details">
          <div class="path">${{escapeHtml(row.path)}}</div>
          <div class="truth"><span>Vérité</span><strong>${{escapeHtml(row.truth || "—")}}</strong></div>
          <div class="predictions">${{modelOrder.map(model => predictionHtml(model, row.models[model])).join("")}}</div>
        </div>
      </div>`;
    }}

    function applyFilters() {{
      filtered = rows.filter(matches);
      index = 0;
      render();
    }}

    Object.values(filters).forEach(element => element.addEventListener("input", applyFilters));
    previous.addEventListener("click", () => {{ if (index > 0) {{ index -= 1; render(); }} }});
    next.addEventListener("click", () => {{ if (index + 1 < filtered.length) {{ index += 1; render(); }} }});
    document.addEventListener("keydown", event => {{
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
      if (event.key === "ArrowLeft") previous.click();
      if (event.key === "ArrowRight") next.click();
    }});
    render();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML model comparison report."
    )
    parser.add_argument("input", type=Path, help="prediction rows JSON")
    parser.add_argument("output", type=Path, help="output HTML file")
    parser.add_argument(
        "--max-images",
        type=int,
        default=200,
        help="maximum number of images embedded (default: 200)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_prediction_rows(args.input)
    report = generate_html(
        rows,
        source_dir=args.input.resolve().parent,
        max_images=args.max_images,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output} ({min(len(rows), args.max_images)} images)")


if __name__ == "__main__":
    main()
