# Codex - Ancient Mesoamerican Glyph Detection

A computer vision project for detecting and classifying ancient Mesoamerican glyphs using YOLOv8.

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/codex.git
   cd codex
   ```

2. **Install dependencies:**
   ```bash
   # Using pip
   pip install -e .
   
   # Or using uv (recommended)
   uv sync
   ```

3. **Download dataset:**
   ```bash
   python setup_data.py
   ```
   This will automatically:
   - Download and extract the image dataset (~700MB)
   - Generate CSV metadata files for analysis

4. **Run the project:**
   ```bash
   python main.py
   # or
   jupyter notebook main.ipynb
   ```

## Project Structure

```
codex/
├── main.py                 # Main training/inference script
├── main.ipynb             # Jupyter notebook for exploration  
├── setup_data.py          # Data download script
├── data_scripts/          # Data processing utilities
├── models/                # Trained model files
├── runs/                  # Training outputs
└── synthetic_dataset/     # Generated synthetic data
```

## Dataset

The project uses a dataset of ancient Mesoamerican glyphs organized into:
- **Elements/**: Individual glyph elements and components
- **Glyphs/**: Complete glyph compositions

The dataset is automatically downloaded on first run and contains:
- ~48,000 image files
- High-quality scans of historical codices
- Labeled annotations for object detection

## Features

- YOLOv8-based glyph detection
- Automated dataset setup and CSV generation
- Synthetic data generation
- Data analysis and visualization
- Training pipeline with validation

## Requirements

- Python ≥3.12
- See `pyproject.toml` for complete dependency list

## Usage

### Training a new model
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(data='path/to/dataset.yaml', epochs=100)
```

### Running inference
```python
model = YOLO('path/to/trained/model.pt')
results = model('path/to/image.jpg')
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license here]

## Acknowledgments

- Ancient Mesoamerican codices and historical sources
- YOLOv8 by Ultralytics
- Contributing researchers and institutions
