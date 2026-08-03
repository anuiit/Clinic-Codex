"""
REST API for the Codex Element Classifier.

Endpoints:
    POST /classify       — classify a single image
    POST /classify-batch  — classify multiple images
    POST /segment        — segment + classify a glyph with MobileSAM
    GET  /classes        — list all 286 element classes

Usage:
    python flask_api.py
    # Then: curl -X POST -F "image=@element.jpg" http://localhost:5000/classify
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codex_model import CodexClassifier

app = Flask(__name__)
clf = CodexClassifier()


@app.route("/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file in request"}), 400
    img = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
    return jsonify(clf.classify(img))


@app.route("/classify-batch", methods=["POST"])
def classify_batch():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No 'images' files in request"}), 400
    images = [np.array(Image.open(io.BytesIO(f.read())).convert("RGB")) for f in files]
    return jsonify(clf.classify_batch(images))


@app.route("/segment", methods=["POST"])
def segment():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file in request"}), 400

    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

    img = np.array(Image.open(io.BytesIO(request.files["image"].read())).convert("RGB"))

    if not hasattr(app, "_sam_gen"):
        checkpoint = str(Path.home() / ".cache/mobile_sam/mobile_sam.pt")
        sam = sam_model_registry["vit_t"](checkpoint=checkpoint)
        sam.eval()
        app._sam_gen = SamAutomaticMaskGenerator(sam, points_per_side=16)

    h, w = img.shape[:2]
    masks = app._sam_gen.generate(img)

    elements = []
    for m in masks:
        if m["area"] < 50 or m["area"] > h * w * 0.85 or m["stability_score"] < 0.8:
            continue
        x, y, bw, bh = m["bbox"]
        crop = img[y:y+bh, x:x+bw].copy()
        crop[~m["segmentation"][y:y+bh, x:x+bw]] = 255
        result = clf.classify(crop)
        elements.append({"bbox": [int(x), int(y), int(bw), int(bh)], **result})

    return jsonify({"num_elements": len(elements), "image_size": [int(w), int(h)], "elements": elements})


@app.route("/classes", methods=["GET"])
def get_classes():
    config_path = Path(__file__).resolve().parents[1] / "codex_model" / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    return jsonify({"num_classes": config["num_classes"], "class_names": config["class_names"]})


if __name__ == "__main__":
    print("Codex Classifier API — http://localhost:5000")
    print("  POST /classify, /classify-batch, /segment")
    print("  GET  /classes")
    app.run(host="0.0.0.0", port=5000, debug=False)
