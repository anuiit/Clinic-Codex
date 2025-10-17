#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify synthetic dataset quality and structure.
Checks YOLO format, visualizes samples, and computes statistics.
"""

import sys
import random
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

# Configure UTF-8 output for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

DATASET_DIR = Path("generated_data")
SAMPLES_OUTPUT = Path("dataset_samples")


def load_config():
    """Load dataset YAML configuration"""
    yaml_path = DATASET_DIR / "dataset.yaml"
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def verify_structure():
    """Verify directory structure"""
    print("\n" + "="*60)
    print("📁 VERIFYING DIRECTORY STRUCTURE")
    print("="*60)

    required_dirs = [
        DATASET_DIR / "train" / "images",
        DATASET_DIR / "train" / "labels",
        DATASET_DIR / "val" / "images",
        DATASET_DIR / "val" / "labels",
    ]

    all_ok = True
    for dir_path in required_dirs:
        if dir_path.exists():
            num_files = len(list(dir_path.glob("*")))
            print(f"✅ {dir_path.relative_to(DATASET_DIR)}: {num_files} files")
        else:
            print(f"❌ {dir_path.relative_to(DATASET_DIR)}: NOT FOUND")
            all_ok = False

    return all_ok


def verify_yaml_config(config):
    """Verify YAML configuration"""
    print("\n" + "="*60)
    print("📄 VERIFYING YAML CONFIGURATION")
    print("="*60)

    required_keys = ['path', 'train', 'val', 'nc', 'names']
    all_ok = True

    for key in required_keys:
        if key in config:
            value = config[key]
            if key == 'names':
                print(f"✅ {key}: {len(value)} classes")
            elif key == 'nc':
                print(f"✅ {key}: {value}")
            else:
                print(f"✅ {key}: {value}")
        else:
            print(f"❌ {key}: MISSING")
            all_ok = False

    # Verify nc matches names length
    if config.get('nc') != len(config.get('names', [])):
        print(f"❌ Mismatch: nc={config['nc']} but names has {len(config['names'])} entries")
        all_ok = False

    return all_ok


def verify_labels(split='train', sample_size=100):
    """Verify label files are in correct YOLO format"""
    print(f"\n" + "="*60)
    print(f"🔍 VERIFYING {split.upper()} LABELS")
    print("="*60)

    labels_dir = DATASET_DIR / split / "labels"
    label_files = list(labels_dir.glob("*.txt"))

    if not label_files:
        print(f"❌ No label files found")
        return False

    print(f"Total label files: {len(label_files)}")

    # Sample random labels
    samples = random.sample(label_files, min(sample_size, len(label_files)))

    errors = []
    total_objects = 0
    class_counts = Counter()
    bbox_sizes = []

    for label_path in samples:
        try:
            with open(label_path, 'r') as f:
                lines = f.read().strip().split('\n')

            for line in lines:
                if not line.strip():
                    continue

                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{label_path.name}: Wrong number of values ({len(parts)})")
                    continue

                class_id, x, y, w, h = map(float, parts)

                # Verify class_id is integer
                if class_id != int(class_id):
                    errors.append(f"{label_path.name}: class_id not integer")

                # Verify normalized values
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    errors.append(f"{label_path.name}: Values not normalized")

                total_objects += 1
                class_counts[int(class_id)] += 1
                bbox_sizes.append((w, h))

        except Exception as e:
            errors.append(f"{label_path.name}: {e}")

    # Report
    print(f"✅ Checked {len(samples)} label files")
    print(f"✅ Total objects: {total_objects}")
    print(f"✅ Average objects per image: {total_objects/len(samples):.1f}")

    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for err in errors[:10]:  # Show first 10 errors
            print(f"   {err}")
    else:
        print(f"✅ No errors found")

    # Bbox size statistics
    if bbox_sizes:
        widths = [w for w, h in bbox_sizes]
        heights = [h for w, h in bbox_sizes]
        print(f"\n📊 Bounding Box Statistics:")
        print(f"   Avg width: {np.mean(widths):.3f} (±{np.std(widths):.3f})")
        print(f"   Avg height: {np.mean(heights):.3f} (±{np.std(heights):.3f})")
        print(f"   Min width: {min(widths):.3f}, Max width: {max(widths):.3f}")
        print(f"   Min height: {min(heights):.3f}, Max height: {max(heights):.3f}")

    # Top classes
    print(f"\n📊 Top 10 Classes:")
    for class_id, count in class_counts.most_common(10):
        print(f"   Class {class_id}: {count} instances")

    return len(errors) == 0


def draw_bbox(image, bbox, label, color=(0, 255, 0)):
    """Draw bounding box on image"""
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Convert normalized to pixels
    x_center, y_center, w, h = bbox
    x1 = int((x_center - w/2) * width)
    y1 = int((y_center - h/2) * height)
    x2 = int((x_center + w/2) * width)
    y2 = int((y_center + h/2) * height)

    # Draw box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    # Draw label
    try:
        # Try to load a font (optional)
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((x1, y1), label, font=font)
    draw.rectangle(text_bbox, fill=color)
    draw.text((x1, y1), label, fill=(255, 255, 255), font=font)


def visualize_samples(config, n_samples=5):
    """Create visualizations of sample images with annotations"""
    print(f"\n" + "="*60)
    print(f"🎨 CREATING VISUALIZATIONS")
    print("="*60)

    SAMPLES_OUTPUT.mkdir(exist_ok=True)

    for split in ['train', 'val']:
        images_dir = DATASET_DIR / split / "images"
        labels_dir = DATASET_DIR / split / "labels"

        image_files = list(images_dir.glob("*.jpg"))
        samples = random.sample(image_files, min(n_samples, len(image_files)))

        print(f"\n{split.upper()} samples:")

        for img_path in samples:
            # Load image
            image = Image.open(img_path).convert("RGB")

            # Load labels
            label_path = labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                print(f"  ⚠️  No labels for {img_path.name}")
                continue

            with open(label_path, 'r') as f:
                lines = f.read().strip().split('\n')

            # Draw bounding boxes
            for line in lines:
                if not line.strip():
                    continue

                parts = line.split()
                class_id = int(float(parts[0]))
                bbox = list(map(float, parts[1:]))

                # Get class name
                class_name = config['names'][class_id]

                # Random color for each class
                random.seed(class_id)
                color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )

                draw_bbox(image, bbox, f"{class_id}: {class_name[:15]}", color)

            # Save visualization
            output_path = SAMPLES_OUTPUT / f"{split}_{img_path.name}"
            image.save(output_path, quality=95)
            print(f"  ✓ {output_path.name}")

    print(f"\n✅ Visualizations saved to {SAMPLES_OUTPUT}/")


def compute_statistics():
    """Compute dataset statistics"""
    print(f"\n" + "="*60)
    print(f"📊 DATASET STATISTICS")
    print("="*60)

    for split in ['train', 'val']:
        images_dir = DATASET_DIR / split / "images"
        labels_dir = DATASET_DIR / split / "labels"

        image_files = list(images_dir.glob("*.jpg"))
        label_files = list(labels_dir.glob("*.txt"))

        print(f"\n{split.upper()}:")
        print(f"  Images: {len(image_files)}")
        print(f"  Labels: {len(label_files)}")

        # Count total objects
        total_objects = 0
        for label_file in label_files:
            with open(label_file, 'r') as f:
                total_objects += len([l for l in f.read().strip().split('\n') if l.strip()])

        print(f"  Total objects: {total_objects}")
        print(f"  Avg objects/image: {total_objects/len(label_files):.2f}")

        # Calculate dataset size
        total_size = sum(f.stat().st_size for f in images_dir.glob("*"))
        print(f"  Total size: {total_size / (1024**2):.1f} MB")


def main():
    """Main verification"""
    print("\n" + "="*60)
    print("🔬 SYNTHETIC DATASET VERIFICATION")
    print("="*60)

    if not DATASET_DIR.exists():
        print(f"❌ Dataset directory not found: {DATASET_DIR}")
        return 1

    # Set seed for reproducible sampling
    random.seed(42)

    try:
        # Load config
        config = load_config()

        # Run verifications
        structure_ok = verify_structure()
        yaml_ok = verify_yaml_config(config)
        train_labels_ok = verify_labels('train', sample_size=100)
        val_labels_ok = verify_labels('val', sample_size=50)

        # Statistics
        compute_statistics()

        # Create visualizations
        visualize_samples(config, n_samples=5)

        # Summary
        print("\n" + "="*60)
        print("📋 VERIFICATION SUMMARY")
        print("="*60)

        checks = {
            "Directory Structure": structure_ok,
            "YAML Configuration": yaml_ok,
            "Train Labels": train_labels_ok,
            "Val Labels": val_labels_ok,
        }

        all_passed = all(checks.values())

        for check_name, passed in checks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {check_name}")

        print("="*60)

        if all_passed:
            print("🎉 ALL CHECKS PASSED!")
            print("✅ Dataset is ready for YOLO training")
        else:
            print("⚠️  SOME CHECKS FAILED")
            print("Please review the errors above")

        print("="*60 + "\n")

        return 0 if all_passed else 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
