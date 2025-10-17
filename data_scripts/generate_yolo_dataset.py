#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate YOLO dataset configuration (data.yaml) for the Codex project.
This script creates a YAML file compatible with Ultralytics YOLOv8 training.
"""

import sys
from pathlib import Path
import yaml
import argparse

# Configure UTF-8 output for Windows compatibility
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def generate_yaml_config(
    data_root: Path,
    output_path: Path,
    include_elements: bool = True,
    include_glyphs: bool = True,
    train_path: str = "train/images",
    val_path: str = "valid/images",
    test_path: str = "test/images"
):
    """
    Generate YOLO dataset YAML configuration file.

    Args:
        data_root: Root directory containing Elements and/or Glyphs folders
        output_path: Path where the YAML file will be saved
        include_elements: Whether to include Elements classes
        include_glyphs: Whether to include Glyphs classes
        train_path: Relative path to training images
        val_path: Relative path to validation images
        test_path: Relative path to test images (optional)
    """

    class_names = []

    # Collect class names from Elements folder
    if include_elements:
        elements_dir = data_root / "Elements"
        if elements_dir.exists():
            element_classes = sorted([
                f"element_{d.name}" for d in elements_dir.iterdir() if d.is_dir()
            ])
            class_names.extend(element_classes)
            print(f"📁 Found {len(element_classes)} element classes")
        else:
            print(f"⚠️  Elements directory not found: {elements_dir}")

    # Collect class names from Glyphs folder
    if include_glyphs:
        glyphs_dir = data_root / "Glyphs"
        if glyphs_dir.exists():
            glyph_classes = sorted([
                f"glyph_{d.name}" for d in glyphs_dir.iterdir() if d.is_dir()
            ])
            class_names.extend(glyph_classes)
            print(f"📁 Found {len(glyph_classes)} glyph classes")
        else:
            print(f"⚠️  Glyphs directory not found: {glyphs_dir}")

    if not class_names:
        raise ValueError("No classes found! Check your data directory structure.")

    # Create YAML data structure
    yaml_data = {
        'path': str(data_root.absolute()),  # Dataset root directory (absolute path)
        'train': train_path,  # Train images (relative to 'path')
        'val': val_path,      # Validation images (relative to 'path')
        'test': test_path,    # Test images (optional, relative to 'path')

        # Number of classes
        'nc': len(class_names),

        # Class names (list)
        'names': class_names
    }

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write YAML file
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

    print(f"\n✅ YAML configuration generated successfully!")
    print(f"📄 File: {output_path.absolute()}")
    print(f"📊 Total classes: {len(class_names)}")
    print(f"📂 Dataset path: {data_root.absolute()}")
    print(f"🚂 Train path: {train_path}")
    print(f"✅ Validation path: {val_path}")
    print(f"🧪 Test path: {test_path}")

    return yaml_data


def generate_simple_yaml(data_root: Path, output_path: Path):
    """
    Generate a simplified YAML for the raw data structure (without train/val split).
    Useful for initial data exploration.
    """

    class_names = []
    class_counts = {}

    # Collect from Elements
    elements_dir = data_root / "Elements"
    if elements_dir.exists():
        for d in sorted(elements_dir.iterdir()):
            if d.is_dir():
                class_name = f"element_{d.name}"
                class_names.append(class_name)
                # Count images
                image_count = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png")))
                class_counts[class_name] = image_count

    # Collect from Glyphs
    glyphs_dir = data_root / "Glyphs"
    if glyphs_dir.exists():
        for d in sorted(glyphs_dir.iterdir()):
            if d.is_dir():
                class_name = f"glyph_{d.name}"
                class_names.append(class_name)
                # Count images
                image_count = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png")))
                class_counts[class_name] = image_count

    yaml_data = {
        'path': str(data_root.absolute()),
        'train': 'images',  # Placeholder
        'val': 'images',    # Placeholder
        'nc': len(class_names),
        'names': class_names,
        '# Note': 'This is a basic configuration. Run split_dataset.py to create train/val splits.'
    }

    # Write YAML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

    print(f"\n✅ Simple YAML configuration generated!")
    print(f"📄 File: {output_path.absolute()}")
    print(f"📊 Total classes: {len(class_names)}")

    # Show top 10 classes by image count
    print(f"\n📈 Top 10 classes by image count:")
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for class_name, count in sorted_classes:
        print(f"   {class_name}: {count} images")

    return yaml_data


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate YOLO dataset YAML configuration for Codex project"
    )
    parser.add_argument(
        '--data-root',
        type=Path,
        default=None,
        help='Root directory containing Elements/Glyphs folders (default: auto-detect)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output YAML file path (default: data/dataset.yaml)'
    )
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Generate simple YAML for raw data (no train/val split required)'
    )
    parser.add_argument(
        '--elements-only',
        action='store_true',
        help='Include only Elements classes'
    )
    parser.add_argument(
        '--glyphs-only',
        action='store_true',
        help='Include only Glyphs classes'
    )

    args = parser.parse_args()

    # Auto-detect data root
    if args.data_root is None:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent

        # Check for data subdirectory
        data_dir = project_root / "data"
        if data_dir.exists() and ((data_dir / "Elements").exists() or (data_dir / "Glyphs").exists()):
            args.data_root = data_dir
        else:
            args.data_root = project_root

    # Set default output path
    if args.output is None:
        args.output = args.data_root / "dataset.yaml"

    print(f"🔍 Data root: {args.data_root.absolute()}")

    # Determine what to include
    include_elements = not args.glyphs_only
    include_glyphs = not args.elements_only

    try:
        if args.simple:
            generate_simple_yaml(args.data_root, args.output)
        else:
            generate_yaml_config(
                args.data_root,
                args.output,
                include_elements=include_elements,
                include_glyphs=include_glyphs
            )
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
