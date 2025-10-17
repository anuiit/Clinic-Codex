#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic Dataset Generator for YOLO Training
Generates realistic synthetic images with element instances for object detection.
Optimized for training efficiency and simplicity.
"""

import sys
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import yaml

# Configure UTF-8 output for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ==================== CONFIGURATION ====================

# Paths
ELEMENTS_DIR = Path("data/Elements")      # Source elements directory
OUTPUT_DIR = Path("generated_data")       # Output directory

# Dataset parameters
N_TRAIN = 1000                            # Number of training images
N_VAL = 200                               # Number of validation images
IMG_SIZE = (640, 640)                     # Image size (optimized for YOLO)
MIN_OBJECTS = 3                           # Min objects per image
MAX_OBJECTS = 8                           # Max objects per image

# Augmentation parameters (realistic but not too aggressive)
ROTATION_RANGE = (-25, 25)                # Rotation in degrees
SCALE_RANGE = (0.4, 1.2)                  # Scale factor
CONTRAST_RANGE = (0.85, 1.15)             # Contrast variation
BRIGHTNESS_RANGE = (0.9, 1.1)             # Brightness variation
BLUR_PROB = 0.3                           # Probability of blur
BLUR_RADIUS = (0.5, 1.5)                  # Blur radius range

# Background parameters
BACKGROUND_COLOR = (240, 240, 240)        # Light gray background
ADD_NOISE = True                          # Add paper-like texture
NOISE_INTENSITY = 8                       # Noise strength

# =======================================================

@dataclass
class BoundingBox:
    """Bounding box in YOLO format (normalized)"""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_yolo_string(self) -> str:
        """Convert to YOLO format string"""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


def setup_output_structure():
    """Create YOLO directory structure"""
    print("🗂️  Setting up output directory structure...")

    # Remove old output if exists
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    # Create directory structure
    (OUTPUT_DIR / "train" / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "val" / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "val" / "labels").mkdir(parents=True, exist_ok=True)

    print(f"✅ Created directory: {OUTPUT_DIR}/")


def load_element_classes() -> Tuple[Dict[str, int], Dict[int, List[Path]]]:
    """
    Load element classes and their image paths.
    Returns: (class_name_to_id, class_id_to_images)
    """
    print(f"📁 Loading element classes from {ELEMENTS_DIR}...")

    if not ELEMENTS_DIR.exists():
        raise FileNotFoundError(f"Elements directory not found: {ELEMENTS_DIR}")

    # Get all class directories
    class_dirs = sorted([d for d in ELEMENTS_DIR.iterdir() if d.is_dir()])

    if not class_dirs:
        raise ValueError(f"No class directories found in {ELEMENTS_DIR}")

    # Create class mappings
    class_to_id = {d.name: idx for idx, d in enumerate(class_dirs)}

    # Load images for each class
    id_to_images = {}
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    for idx, class_dir in enumerate(class_dirs):
        images = [
            img for img in class_dir.iterdir()
            if img.suffix.lower() in valid_extensions
        ]

        if images:
            id_to_images[idx] = images
            print(f"  ✓ {class_dir.name}: {len(images)} images")
        else:
            print(f"  ⚠️  {class_dir.name}: No images found (skipping)")

    print(f"✅ Loaded {len(id_to_images)} classes with images")
    return class_to_id, id_to_images


def create_background(size: Tuple[int, int]) -> Image.Image:
    """Create a paper-like background"""
    width, height = size

    # Create base background
    background = Image.new("RGB", size, BACKGROUND_COLOR)

    if ADD_NOISE:
        # Add paper texture noise
        arr = np.array(background, dtype=np.float32)
        noise = np.random.normal(0, NOISE_INTENSITY, (height, width, 3))
        arr = np.clip(arr + noise, 0, 255)
        background = Image.fromarray(arr.astype(np.uint8))

    return background


def augment_element(image: Image.Image) -> Image.Image:
    """Apply augmentations to an element image"""
    # Convert to RGBA if needed
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Random scale
    scale = random.uniform(*SCALE_RANGE)
    new_w = max(10, int(image.width * scale))
    new_h = max(10, int(image.height * scale))
    image = image.resize((new_w, new_h), Image.LANCZOS)

    # Random rotation (with expand to keep full object)
    angle = random.uniform(*ROTATION_RANGE)
    image = image.rotate(angle, expand=True, resample=Image.BICUBIC)

    # Random contrast and brightness
    contrast = random.uniform(*CONTRAST_RANGE)
    brightness = random.uniform(*BRIGHTNESS_RANGE)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Brightness(image).enhance(brightness)

    # Random blur
    if random.random() < BLUR_PROB:
        radius = random.uniform(*BLUR_RADIUS)
        image = image.filter(ImageFilter.GaussianBlur(radius))

    return image


def paste_element(
    canvas: Image.Image,
    element: Image.Image,
    position: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Paste element on canvas and return bounding box (x1, y1, x2, y2).
    """
    x, y = position

    # Ensure RGBA
    if element.mode != "RGBA":
        element = element.convert("RGBA")

    # Create temporary layer
    temp = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    temp.paste(element, (x, y), element)

    # Composite onto canvas
    canvas_rgba = canvas.convert("RGBA")
    canvas_rgba = Image.alpha_composite(canvas_rgba, temp)
    canvas.paste(canvas_rgba.convert("RGB"))

    # Calculate bounding box from alpha channel
    alpha = np.array(temp.split()[-1])
    ys, xs = np.where(alpha > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None  # No visible pixels

    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())

    return (x1, y1, x2, y2)


def bbox_to_yolo(
    bbox: Tuple[int, int, int, int],
    img_size: Tuple[int, int],
    class_id: int
) -> BoundingBox:
    """Convert pixel bbox to YOLO normalized format"""
    x1, y1, x2, y2 = bbox
    img_w, img_h = img_size

    # Calculate center and dimensions
    width = x2 - x1
    height = y2 - y1
    x_center = x1 + width / 2.0
    y_center = y1 + height / 2.0

    # Normalize
    x_center /= img_w
    y_center /= img_h
    width /= img_w
    height /= img_h

    return BoundingBox(class_id, x_center, y_center, width, height)


def generate_image(
    id_to_images: Dict[int, List[Path]],
    image_size: Tuple[int, int]
) -> Tuple[Image.Image, List[BoundingBox]]:
    """Generate a single synthetic image with annotations"""

    # Create background
    canvas = create_background(image_size)
    bboxes = []

    # Number of objects to place
    n_objects = random.randint(MIN_OBJECTS, MAX_OBJECTS)

    # Get available classes
    available_classes = list(id_to_images.keys())

    for _ in range(n_objects):
        # Select random class
        class_id = random.choice(available_classes)

        # Select random image from class
        element_path = random.choice(id_to_images[class_id])

        # Load and augment element
        element = Image.open(element_path)
        element = augment_element(element)

        # Random position (ensure element fits)
        max_x = max(0, image_size[0] - element.width)
        max_y = max(0, image_size[1] - element.height)

        if max_x <= 0 or max_y <= 0:
            continue  # Element too large, skip

        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        # Paste element and get bbox
        bbox_pixels = paste_element(canvas, element, (x, y))

        if bbox_pixels is None:
            continue  # No visible pixels, skip

        # Check bbox is valid (not too small)
        x1, y1, x2, y2 = bbox_pixels
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            continue  # Too small

        # Convert to YOLO format
        bbox_yolo = bbox_to_yolo(bbox_pixels, image_size, class_id)
        bboxes.append(bbox_yolo)

    return canvas, bboxes


def generate_dataset(
    id_to_images: Dict[int, List[Path]],
    n_images: int,
    split: str
) -> None:
    """Generate dataset for a specific split (train/val)"""

    print(f"\n🎨 Generating {n_images} {split} images...")

    images_dir = OUTPUT_DIR / split / "images"
    labels_dir = OUTPUT_DIR / split / "labels"

    for i in range(n_images):
        # Generate image
        image, bboxes = generate_image(id_to_images, IMG_SIZE)

        # Skip if no valid objects
        if not bboxes:
            print(f"  ⚠️  Image {i+1}: No valid objects, regenerating...")
            # Try again
            image, bboxes = generate_image(id_to_images, IMG_SIZE)

        # Save image
        image_name = f"{i+1:06d}.jpg"
        image.save(images_dir / image_name, quality=90, optimize=True)

        # Save labels
        label_name = f"{i+1:06d}.txt"
        label_lines = [bbox.to_yolo_string() for bbox in bboxes]
        (labels_dir / label_name).write_text("\n".join(label_lines), encoding='utf-8')

        # Progress
        if (i + 1) % 50 == 0 or (i + 1) == n_images:
            print(f"  ✓ {i+1}/{n_images} images generated")


def create_yaml_config(class_to_id: Dict[str, int]) -> None:
    """Create YOLO dataset configuration file"""

    print("\n📄 Creating dataset.yaml configuration...")

    # Get class names in order
    class_names = [name for name, _ in sorted(class_to_id.items(), key=lambda x: x[1])]

    yaml_data = {
        'path': str(OUTPUT_DIR.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(class_names),
        'names': class_names
    }

    yaml_path = OUTPUT_DIR / "dataset.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True)

    print(f"✅ Created {yaml_path}")
    print(f"   Classes: {len(class_names)}")
    print(f"   Train images: train/images/")
    print(f"   Val images: val/images/")


def print_summary(class_to_id: Dict[str, int]):
    """Print generation summary"""

    print("\n" + "="*60)
    print("📊 DATASET GENERATION SUMMARY")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print(f"Image size: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    print(f"Training images: {N_TRAIN}")
    print(f"Validation images: {N_VAL}")
    print(f"Total images: {N_TRAIN + N_VAL}")
    print(f"Classes: {len(class_to_id)}")
    print(f"Objects per image: {MIN_OBJECTS}-{MAX_OBJECTS}")
    print("="*60)

    # Check file counts
    train_images = len(list((OUTPUT_DIR / "train" / "images").glob("*.jpg")))
    train_labels = len(list((OUTPUT_DIR / "train" / "labels").glob("*.txt")))
    val_images = len(list((OUTPUT_DIR / "val" / "images").glob("*.jpg")))
    val_labels = len(list((OUTPUT_DIR / "val" / "labels").glob("*.txt")))

    print(f"\n✅ Generated files:")
    print(f"   Train: {train_images} images, {train_labels} labels")
    print(f"   Val: {val_images} images, {val_labels} labels")

    # Calculate dataset size
    total_size = sum(
        f.stat().st_size
        for f in OUTPUT_DIR.rglob("*")
        if f.is_file()
    )
    print(f"   Total size: {total_size / (1024**2):.1f} MB")

    print("\n🎉 Dataset generation complete!")
    print(f"📂 Ready to train with: {OUTPUT_DIR / 'dataset.yaml'}")


def main():
    """Main entry point"""

    print("\n" + "="*60)
    print("🚀 SYNTHETIC DATASET GENERATOR FOR YOLO")
    print("="*60)

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    try:
        # Setup
        setup_output_structure()

        # Load classes
        class_to_id, id_to_images = load_element_classes()

        # Generate training set
        generate_dataset(id_to_images, N_TRAIN, "train")

        # Generate validation set
        generate_dataset(id_to_images, N_VAL, "val")

        # Create YAML config
        create_yaml_config(class_to_id)

        # Print summary
        print_summary(class_to_id)

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
