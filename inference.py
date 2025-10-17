#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Inference Script for Glyph Detection
Run predictions on images using your trained model.
"""

import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2

# Configure UTF-8 output for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def run_inference(model_path, image_path, conf_threshold=0.25, save_output=True):
    """
    Run inference on an image
    
    Args:
        model_path: Path to trained YOLO model (.pt file)
        image_path: Path to input image
        conf_threshold: Confidence threshold for detections
        save_output: Whether to save the annotated output image
    """
    
    # Load model
    print(f"📦 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Run inference
    print(f"🔍 Running inference on: {image_path}")
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=save_output,
        project='runs/predict',
        name='inference',
        exist_ok=True
    )
    
    # Display results
    for i, result in enumerate(results):
        print(f"\n📊 Results for image {i+1}:")
        print(f"   Detected {len(result.boxes)} objects")
        
        # Print detections
        if len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                print(f"   - {class_name}: {confidence:.2%}")
        else:
            print("   No objects detected")
        
        if save_output:
            print(f"\n💾 Output saved to: runs/predict/inference/")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run YOLO inference on images')
    parser.add_argument('--model', '-m', type=str, 
                       default='runs/train/glyph_detection/weights/best.pt',
                       help='Path to trained model (default: best.pt from training)')
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--conf', '-c', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save output images')
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        print("   Please train a model first: python train_yolo.py")
        sys.exit(1)
    
    # Check if image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Error: Image not found at {image_path}")
        sys.exit(1)
    
    # Run inference
    print("=" * 80)
    print("YOLO Inference - Glyph Detection")
    print("=" * 80)
    
    run_inference(
        model_path=str(model_path),
        image_path=str(image_path),
        conf_threshold=args.conf,
        save_output=not args.no_save
    )
    
    print("\n✅ Inference completed!")


if __name__ == "__main__":
    main()
