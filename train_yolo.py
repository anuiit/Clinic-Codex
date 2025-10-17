#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Model Training Script for Ancient Mesoamerican Glyph Detection
Trains a YOLOv8 model on the synthetic dataset.
"""

import sys
from pathlib import Path
from ultralytics import YOLO

# Configure UTF-8 output for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ==================== CONFIGURATION ====================

# Paths
DATASET_YAML = Path("generated_data/dataset.yaml")
PRETRAINED_MODEL = Path("yolov8n.pt")  # YOLOv8 nano (fastest)
# Other options: yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large)

# Training parameters
EPOCHS = 100                    # Number of training epochs
BATCH_SIZE = 16                 # Batch size (adjust based on GPU memory)
IMAGE_SIZE = 640                # Image size for training
DEVICE = 0                      # GPU device (0 for first GPU, 'cpu' for CPU)
PATIENCE = 20                   # Early stopping patience
WORKERS = 8                     # Number of data loading workers

# Training options
SAVE_PERIOD = 10                # Save checkpoint every N epochs
PRETRAINED = True               # Use pretrained weights
OPTIMIZER = 'auto'              # Optimizer (auto, SGD, Adam, AdamW)
LR0 = 0.01                      # Initial learning rate
LRF = 0.01                      # Final learning rate (lr0 * lrf)

# Augmentation (YOLO handles this automatically, but you can customize)
AUG_SETTINGS = {
    'hsv_h': 0.015,             # Image HSV-Hue augmentation
    'hsv_s': 0.7,               # Image HSV-Saturation augmentation
    'hsv_v': 0.4,               # Image HSV-Value augmentation
    'degrees': 0.0,             # Image rotation (+/- deg)
    'translate': 0.1,           # Image translation (+/- fraction)
    'scale': 0.5,               # Image scale (+/- gain)
    'shear': 0.0,               # Image shear (+/- deg)
    'perspective': 0.0,         # Image perspective (+/- fraction)
    'flipud': 0.0,              # Image flip up-down (probability)
    'fliplr': 0.5,              # Image flip left-right (probability)
    'mosaic': 1.0,              # Image mosaic (probability)
    'mixup': 0.0,               # Image mixup (probability)
}

# ==================== TRAINING ====================

def train_model():
    """Train YOLOv8 model on the synthetic dataset"""
    
    print("=" * 80)
    print("YOLO Model Training - Ancient Mesoamerican Glyph Detection")
    print("=" * 80)
    
    # Verify dataset exists
    if not DATASET_YAML.exists():
        print(f"❌ Error: Dataset YAML not found at {DATASET_YAML}")
        print("   Please run: python generate_synthetic_dataset.py")
        sys.exit(1)
    
    # Verify pretrained model exists (optional)
    if PRETRAINED and not PRETRAINED_MODEL.exists():
        print(f"⚠️  Warning: Pretrained model not found at {PRETRAINED_MODEL}")
        print("   Will download automatically from Ultralytics...")
    
    # Load model
    print(f"\n📦 Loading model: {PRETRAINED_MODEL}")
    model = YOLO(str(PRETRAINED_MODEL))
    
    # Display training configuration
    print("\n⚙️  Training Configuration:")
    print(f"   Dataset: {DATASET_YAML}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Image Size: {IMAGE_SIZE}")
    print(f"   Device: {DEVICE}")
    print(f"   Optimizer: {OPTIMIZER}")
    print(f"   Learning Rate: {LR0} → {LRF}")
    print(f"   Early Stopping Patience: {PATIENCE}")
    
    # Train the model
    print("\n🚀 Starting training...\n")
    
    try:
        results = model.train(
            data=str(DATASET_YAML),
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            patience=PATIENCE,
            save_period=SAVE_PERIOD,
            pretrained=PRETRAINED,
            optimizer=OPTIMIZER,
            lr0=LR0,
            lrf=LRF,
            workers=WORKERS,
            project='runs/train',        # Save to runs/train/
            name='glyph_detection',      # Experiment name
            exist_ok=False,              # Don't overwrite existing
            verbose=True,                # Verbose output
            **AUG_SETTINGS               # Augmentation settings
        )
        
        print("\n" + "=" * 80)
        print("✅ Training completed successfully!")
        print("=" * 80)
        
        # Display results
        print("\n📊 Training Results:")
        print(f"   Best weights: {model.trainer.best}")
        print(f"   Last weights: {model.trainer.last}")
        print(f"   Results saved to: {model.trainer.save_dir}")
        
        # Validation results
        print("\n📈 Final Metrics:")
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"   mAP50: {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
            print(f"   mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
            print(f"   Precision: {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
            print(f"   Recall: {metrics.get('metrics/recall(B)', 'N/A'):.4f}")
        
        print("\n💡 Next Steps:")
        print("   1. View training plots: Check runs/train/glyph_detection/")
        print("   2. Test your model: python test_model.py")
        print("   3. Run inference: python inference.py --image path/to/image.jpg")
        
        return results
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point"""
    train_model()


if __name__ == "__main__":
    main()
