# YOLO Training Guide - Clinic-Codex

## Quick Start

### 1. Train a Model

```bash
python train_yolo.py
```

This will:
- Load the YOLOv8n pretrained model (`yolov8n.pt`)
- Train on your synthetic dataset (`generated_data/`)
- Save checkpoints to `runs/train/glyph_detection/`
- Display training progress and metrics

**Training will take:**
- ~30-60 minutes on a modern GPU (NVIDIA RTX 3060 or better)
- Several hours on CPU (not recommended)

### 2. Run Inference

After training, test your model on an image:

```bash
python inference.py --image path/to/your/image.jpg
```

View results in `runs/predict/inference/`

## Detailed Instructions

### Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -e .
   ```
   or with uv:
   ```bash
   uv sync
   ```

2. **Generate dataset (if not already done):**
   ```bash
   python generate_synthetic_dataset.py
   ```

3. **Verify dataset:**
   ```bash
   python verify_synthetic_dataset.py
   ```

### Training Configuration

Edit `train_yolo.py` to customize training:

```python
# Model selection
PRETRAINED_MODEL = Path("yolov8n.pt")  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x

# Training parameters
EPOCHS = 100              # More epochs = better training (but slower)
BATCH_SIZE = 16           # Reduce if GPU memory is insufficient
IMAGE_SIZE = 640          # YOLO input size (640 is standard)
DEVICE = 0                # GPU device (0 = first GPU, 'cpu' = CPU)
PATIENCE = 20             # Early stopping after N epochs without improvement
```

**Model sizes:**
- `yolov8n.pt` - Nano (fastest, smallest, ~3.2M parameters)
- `yolov8s.pt` - Small (~11.2M parameters)
- `yolov8m.pt` - Medium (~25.9M parameters)
- `yolov8l.pt` - Large (~43.7M parameters)
- `yolov8x.pt` - Extra Large (largest, slowest, best accuracy)

### Monitoring Training

**During training, you'll see:**
- Loss values (box_loss, cls_loss, dfl_loss) - lower is better
- Metrics (precision, recall, mAP50, mAP50-95) - higher is better
- Progress bar with ETA

**TensorBoard (optional):**
```bash
tensorboard --logdir runs/train
```

### Understanding Output

After training completes, check these directories:

```
runs/train/glyph_detection/
├── weights/
│   ├── best.pt          # Best model (use this for inference!)
│   └── last.pt          # Last epoch model
├── results.png          # Training metrics plot
├── confusion_matrix.png # Class confusion matrix
├── F1_curve.png         # F1 score curve
├── P_curve.png          # Precision curve
├── R_curve.png          # Recall curve
├── PR_curve.png         # Precision-Recall curve
└── args.yaml            # Training arguments
```

### Key Metrics Explained

- **mAP50**: Mean Average Precision at 0.5 IoU threshold (primary metric)
- **mAP50-95**: Mean Average Precision averaged over IoU 0.5 to 0.95
- **Precision**: Accuracy of positive predictions
- **Recall**: Ability to find all positive samples
- **box_loss**: Bounding box regression loss
- **cls_loss**: Classification loss

**Good targets for this dataset:**
- mAP50 > 0.7 (70%+)
- mAP50-95 > 0.4 (40%+)
- Precision & Recall > 0.6 (60%+)

### Common Issues & Solutions

#### 1. Out of Memory Error
```
CUDA out of memory
```
**Solution:** Reduce `BATCH_SIZE` in `train_yolo.py`:
```python
BATCH_SIZE = 8  # or even 4
```

#### 2. Slow Training (No GPU)
```
Device: cpu
```
**Solution:** 
- Install CUDA and PyTorch with GPU support
- Or use Google Colab with free GPU
- Or reduce epochs and use smaller model (yolov8n)

#### 3. Poor Performance (Low mAP)
**Solutions:**
- Increase `EPOCHS` (try 200-300)
- Generate more training data
- Adjust learning rate
- Use larger model (yolov8s or yolov8m)

#### 4. Dataset Not Found
```
❌ Error: Dataset YAML not found
```
**Solution:**
```bash
python generate_synthetic_dataset.py
```

### Advanced: Fine-tuning Parameters

#### Learning Rate
```python
LR0 = 0.01      # Initial learning rate (default)
LRF = 0.01      # Final learning rate multiplier

# For fine-tuning a model: use lower LR
LR0 = 0.001
```

#### Data Augmentation
```python
AUG_SETTINGS = {
    'mosaic': 1.0,        # Mosaic augmentation (combines 4 images)
    'mixup': 0.0,         # MixUp augmentation
    'degrees': 10.0,      # Rotation augmentation
    'translate': 0.1,     # Translation augmentation
    'scale': 0.5,         # Scale augmentation
    'fliplr': 0.5,        # Horizontal flip
}
```

#### Optimizer
```python
OPTIMIZER = 'AdamW'  # Options: SGD, Adam, AdamW, NAdam, RAdam
```

### Inference Options

#### Basic Inference
```bash
python inference.py --image test_image.jpg
```

#### With Custom Model
```bash
python inference.py --model runs/train/glyph_detection2/weights/best.pt --image test.jpg
```

#### Adjust Confidence Threshold
```bash
python inference.py --image test.jpg --conf 0.5  # Only show detections > 50% confidence
```

#### Batch Inference (Directory)
```bash
python inference.py --image path/to/image_folder/
```

### Using the Model Programmatically

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/train/glyph_detection/weights/best.pt')

# Run inference
results = model('path/to/image.jpg')

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        class_name = model.names[class_id]
        
        print(f"{class_name}: {confidence:.2%} at {bbox}")
```

### Export Model (for deployment)

```python
from ultralytics import YOLO

model = YOLO('runs/train/glyph_detection/weights/best.pt')

# Export to different formats
model.export(format='onnx')      # ONNX
model.export(format='torchscript')  # TorchScript
model.export(format='coreml')    # CoreML (iOS)
model.export(format='tflite')    # TensorFlow Lite
```

## Next Steps

1. **Train your first model:**
   ```bash
   python train_yolo.py
   ```

2. **Monitor training in `runs/train/glyph_detection/`**

3. **Test on validation images:**
   ```bash
   python inference.py --image generated_data/val/images/val_0001.png
   ```

4. **Experiment with different models and parameters**

5. **Deploy your model in production**

## Resources

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLO Training Tips](https://github.com/ultralytics/ultralytics/wiki/Tips-for-Best-Training-Results)
- [Understanding mAP](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)

## Troubleshooting

For issues, check:
1. GPU drivers are up to date
2. PyTorch with CUDA is installed: `python -c "import torch; print(torch.cuda.is_available())"`
3. Dataset is generated: `ls generated_data/train/images/`
4. Enough disk space for training outputs (~500MB)

Good luck with your training! 🚀
