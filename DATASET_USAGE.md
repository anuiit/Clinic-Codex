# Dataset Usage Guide

This guide explains the generated dataset files and how to use them for training and analysis.

## Generated Files

### CSV Files (in `data/` directory)

1. **`image_data_alljpg.csv`** - Complete dataset information
   - 14,999 rows (one per image)
   - 15 columns including metadata, dimensions, file sizes
   - Use for: Complete dataset analysis, statistics, data exploration

2. **`image_data_summary_v2.csv`** - Streamlined version
   - 14,999 rows
   - 12 columns (without file size details)
   - Use for: Quick analysis and visualization

3. **`image_data_summary.csv`** - Legacy format
   - Maintained for backward compatibility
   - Same as `image_data_alljpg.csv`

### YAML Configuration

**`dataset.yaml`** - YOLOv8 Training Configuration
- Path: Points to the dataset root directory
- Classes: 614 total classes
  - 303 element classes (from `Elements/` folder)
  - 311 glyph classes (from `Glyphs/` folder)
- Train/Val: Currently set to 'images' (placeholder)

## Dataset Statistics

- **Total Images**: 14,999
  - Elements: 4,623 images (303 categories)
  - Glyphs: 10,376 images (311 categories)
- **Total Categories**: 614
- **Average Image Size**: 138x139 pixels
- **Average File Size**: 13.63 KB
- **Format**: All JPG images

## Using the CSV Files

### Python (Pandas)

```python
import pandas as pd

# Load the complete dataset
df = pd.read_csv('data/image_data_alljpg.csv')

# Basic statistics
print(f"Total images: {len(df)}")
print(f"Categories: {df['full_category'].nunique()}")

# Filter by type
elements = df[df['folder_type'] == 'Elements']
glyphs = df[df['folder_type'] == 'Glyphs']

# Find largest images
largest = df.nlargest(10, 'total_pixels')

# Group by category
by_category = df.groupby('full_category').agg({
    'file_name': 'count',
    'width': 'mean',
    'height': 'mean'
})
```

### Analysis Examples

```python
# Top 10 categories by image count
top_categories = df['full_category'].value_counts().head(10)

# Average dimensions by folder type
df.groupby('folder_type')[['width', 'height']].mean()

# File size distribution
df['file_size_kb'].describe()
```

## Using the YAML for YOLOv8 Training

### Current Configuration

The `dataset.yaml` file is ready but needs proper train/val split. Two options:

#### Option 1: Use existing split script (if you have annotated data)

```bash
# If you have labels in YOLO format
python data_scripts/split_dataset.py
```

#### Option 2: Manual Training Setup

```python
from ultralytics import YOLO

# Load a pre-trained model
model = YOLO('yolov8n.pt')

# Train (you'll need to create train/val directories with images and labels)
results = model.train(
    data='data/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

### Important Notes

1. **Annotations Required**: For YOLO training, you need:
   - Images organized in `train/images/` and `val/images/`
   - Corresponding labels in `train/labels/` and `val/labels/`

2. **Current Status**: The YAML points to raw data structure. You need to:
   - Create bounding box annotations for your images
   - Split data into train/validation sets
   - Organize according to YOLO format

## Running the Scripts

### Generate CSV Files

```bash
# Generate all CSV metadata files
uv run python data_scripts/data_processing.py
```

### Generate YAML Configuration

```bash
# Generate simple YAML (current structure)
uv run python data_scripts/generate_yolo_dataset.py --simple

# Generate full YAML (requires train/val split)
uv run python data_scripts/generate_yolo_dataset.py

# Elements only
uv run python data_scripts/generate_yolo_dataset.py --elements-only

# Glyphs only
uv run python data_scripts/generate_yolo_dataset.py --glyphs-only
```

### Verify Setup

```bash
# Run comprehensive verification
uv run python verify_setup.py
```

## Class Naming Convention

Classes are prefixed to distinguish elements from glyphs:

- **Elements**: `element_<code>-<name>`
  - Example: `element_0015-toloa`

- **Glyphs**: `glyph_<code>-<name>`
  - Example: `glyph_0549-yaotl`

The code represents the number of appearances in the dataset.

## Top Categories by Image Count

1. **glyph_0549-yaotl**: 549 images
2. **glyph_0379-mixcohuatl**: 379 images
3. **glyph_0257-centecpanpixqui**: 257 images
4. **glyph_0247-quiyauh**: 247 images
5. **glyph_0246-cohuatl**: 246 images

## Next Steps

1. **Data Exploration**: Use the CSV files to understand your dataset
2. **Annotation**: Create bounding box annotations if needed
3. **Train/Val Split**: Split your data for training
4. **Model Training**: Use the YAML file with YOLOv8
5. **Evaluation**: Test and validate your model

## Troubleshooting

### Re-generate CSV files
```bash
uv run python data_scripts/data_processing.py
```

### Re-generate YAML
```bash
uv run python data_scripts/generate_yolo_dataset.py --simple
```

### Verify everything works
```bash
uv run python verify_setup.py
```

## File Locations

```
data/
├── Elements/               # Element images organized by category
├── Glyphs/                 # Glyph images organized by category
├── dataset.yaml            # YOLO training configuration
├── image_data_alljpg.csv   # Complete dataset metadata
├── image_data_summary_v2.csv
└── image_data_summary.csv

data_scripts/
├── data_processing.py           # Generate CSV files
├── generate_yolo_dataset.py     # Generate YAML configuration
└── (other scripts...)

verify_setup.py             # Comprehensive verification script
```

## Support

For issues or questions:
1. Run `verify_setup.py` to check your setup
2. Check the error messages
3. Re-run the appropriate generation script
4. Refer to YOLOv8 documentation for training-specific questions
