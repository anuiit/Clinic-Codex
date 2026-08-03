# Codex Element Classifier — Full Integration Package

## What's In This Package

Everything you need to integrate the codex element recognition model into an application.

```
frontend_integration/
├── codex_model/              <- STANDALONE ML model (copy this into your project)
│   ├── classifier.py         <- CodexClassifier class
│   ├── config.json           <- 286 class names + model config
│   └── weights/              <- pretrained weights (prototypes + projection head)
│
├── codex_pipeline/           <- FULL PIPELINE (segmentation + classification + export)
│   ├── inference/            <- InferenceEngine, spatial grouping, JSON/CSV/PNG export
│   ├── segmentation/         <- MobileSAM wrapper for automatic segmentation
│   ├── scripts/              <- CLI tools (infer_glyph.py, infer_page.py, etc.)
│   └── config/default.yaml   <- all pipeline parameters
│
├── prototypes/               <- Exported model knowledge (286 class prototypes)
│   └── prototypes.pt
│
├── data/                     <- SAMPLE DATA for testing
│   ├── glyphs_sample/        <- 30 glyph images (5 per class, 6 classes)
│   └── elements_sample/      <- 30 element images (3 per class, 10 classes)
│
├── examples/                 <- CODE EXAMPLES
│   ├── classify_single.py    <- Classify one image
│   ├── classify_batch.py     <- Classify a folder of images
│   ├── flask_api.py          <- REST API (4 endpoints)
│   ├── segment_glyph.py      <- MobileSAM glyph decomposition
│   ├── infer_glyph.py        <- Full glyph inference script
│   ├── test_sam_on_glyphs.py <- SAM segmentation test with visualization
│   └── sample_outputs/       <- Example visualizations (SAM decompositions)
│
├── requirements.txt          <- All Python dependencies
└── default.yaml              <- Pipeline configuration
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Classify a single element

```python
from codex_model import CodexClassifier

clf = CodexClassifier()
result = clf.classify("data/elements_sample/0015-cacahuatl/03_04_22-27.bmp")
print(result)
# {"class_name": "cacahuatl", "confidence": 0.87, "top_k": [...]}
```

### 3. Decompose a glyph into elements (with MobileSAM)

```python
from codex_model import CodexClassifier
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import numpy as np

# Load models
clf = CodexClassifier()
sam = sam_model_registry["vit_t"](checkpoint="~/.cache/mobile_sam/mobile_sam.pt")
sam.eval()
mask_gen = SamAutomaticMaskGenerator(sam, points_per_side=16)

# Segment + classify
image = np.array(Image.open("data/glyphs_sample/atl-glyph/026r_a_07-2.jpg").convert("RGB"))
masks = mask_gen.generate(image)

for mask in masks:
    x, y, w, h = mask["bbox"]
    crop = image[y:y+h, x:x+w].copy()
    crop[~mask["segmentation"][y:y+h, x:x+w]] = 255
    result = clf.classify(crop)
    if not result["rejected"]:
        print(f"  {result['class_name']} (confidence={result['confidence']:.3f}) at [{x},{y},{w},{h}]")
```

### 4. Run the Flask API

```bash
python examples/flask_api.py
# Server starts on http://localhost:5000

# Classify:
curl -X POST -F "image=@data/elements_sample/0015-cacahuatl/03_04_22-27.bmp" http://localhost:5000/classify

# Segment a glyph:
curl -X POST -F "image=@data/glyphs_sample/atl-glyph/026r_a_07-2.jpg" http://localhost:5000/segment

# List all classes:
curl http://localhost:5000/classes
```

---

## API Reference

### CodexClassifier

```python
clf = CodexClassifier(model_dir=None)  # auto-detects weights

# Single image
result = clf.classify(image)  # PIL Image, numpy array, or file path
# Returns: {"class_name": "atl", "class_label": 282, "confidence": 0.699,
#           "rejected": false, "top_k": [...]}

# Batch
results = clf.classify_batch([img1, img2, img3])  # list of results
```

### Flask API Endpoints

| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| `/classify` | POST | `image` file | `{"class_name", "confidence", ...}` |
| `/classify-batch` | POST | `images` files | `[{"class_name", ...}, ...]` |
| `/segment` | POST | `image` file | `{"num_elements", "elements": [...]}` |
| `/classes` | GET | - | `{"num_classes": 286, "class_names": [...]}` |

### Response Format

```json
{
    "class_name": "atl",
    "class_label": 282,
    "confidence": 0.699,
    "rejected": false,
    "top_k": [
        {"class_name": "atl", "confidence": 0.699},
        {"class_name": "tlalli", "confidence": 0.538},
        {"class_name": "tlacuilolli", "confidence": 0.512}
    ]
}
```

- `confidence` < 0.35 = `rejected: true` = element unknown, needs human review
- `top_k` = 3 best candidates ranked by confidence

---

## Performance

| Operation | CPU (Apple M1) | GPU |
|-----------|---------------|-----|
| Model load | ~1s | ~0.5s |
| Single classify | ~100ms | ~20ms |
| Batch of 32 | ~3s | ~0.5s |
| MobileSAM segment | ~5s | ~0.3s |
| Memory | ~500MB | ~1GB |

---

## For Frontend Integration

### JavaScript/TypeScript (via API)

```javascript
// Classify an element on hover
async function classifyElement(imageBlob) {
    const formData = new FormData();
    formData.append("image", imageBlob);
    
    const response = await fetch("http://localhost:5000/classify", {
        method: "POST",
        body: formData
    });
    
    const result = await response.json();
    return {
        name: result.class_name,        // "atl"
        confidence: result.confidence,   // 0.699
        isKnown: !result.rejected        // true
    };
}

// Decompose a glyph into elements
async function segmentGlyph(imageBlob) {
    const formData = new FormData();
    formData.append("image", imageBlob);
    
    const response = await fetch("http://localhost:5000/segment", {
        method: "POST",
        body: formData
    });
    
    const result = await response.json();
    // result.elements = [{class_name, confidence, bbox: [x,y,w,h]}, ...]
    return result.elements;
}
```

---

## Sample Data Included

### Glyphs (30 images)
6 classes, 5 images each: atl (water), calli (house), cohuatl (snake), cuauhtli (eagle), tochtli (rabbit), pantli (flag)

### Elements (30 images)
10 classes, 3 images each: sample training elements in BMP format

### Sample Outputs
SAM decomposition visualizations showing glyphs segmented into elements with bounding boxes and classifications.

---

## Notes

- **No retraining needed.** Everything is pretrained.
- **286 Nahuatl classes** recognized (full list in `codex_model/config.json`)
- **MobileSAM checkpoint** must be downloaded separately (~39MB): stored at `~/.cache/mobile_sam/mobile_sam.pt`
- **Thread-safe:** create one `CodexClassifier` instance, reuse across requests
