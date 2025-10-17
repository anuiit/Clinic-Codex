#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive verification script for the Codex dataset setup.
Checks CSV files, YAML configuration, and data integrity.
"""

import sys
from pathlib import Path
import pandas as pd
import yaml

# Configure UTF-8 output for Windows compatibility
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def verify_csv_files(data_root: Path):
    """Verify that CSV files exist and have correct structure."""
    print("\n" + "="*60)
    print("📊 VERIFYING CSV FILES")
    print("="*60)

    csv_files = {
        'all_data': data_root / 'image_data_alljpg.csv',
        'summary_v2': data_root / 'image_data_summary_v2.csv',
        'summary': data_root / 'image_data_summary.csv'
    }

    results = {}
    for name, csv_path in csv_files.items():
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                results[name] = {
                    'exists': True,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'size_mb': csv_path.stat().st_size / (1024 * 1024)
                }
                print(f"✅ {csv_path.name}")
                print(f"   Rows: {len(df):,}")
                print(f"   Columns: {len(df.columns)}")
                print(f"   Size: {results[name]['size_mb']:.2f} MB")
            except Exception as e:
                results[name] = {'exists': True, 'error': str(e)}
                print(f"❌ {csv_path.name} - Error: {e}")
        else:
            results[name] = {'exists': False}
            print(f"❌ {csv_path.name} - NOT FOUND")

    return results


def verify_yaml_config(data_root: Path):
    """Verify YAML dataset configuration."""
    print("\n" + "="*60)
    print("📄 VERIFYING YAML CONFIGURATION")
    print("="*60)

    yaml_path = data_root / 'dataset.yaml'

    if not yaml_path.exists():
        print(f"❌ {yaml_path.name} - NOT FOUND")
        return {'exists': False}

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        print(f"✅ {yaml_path.name}")
        print(f"   Path: {config.get('path', 'N/A')}")
        print(f"   Train: {config.get('train', 'N/A')}")
        print(f"   Val: {config.get('val', 'N/A')}")
        print(f"   Number of classes: {config.get('nc', 'N/A')}")

        # Show sample classes
        names = config.get('names', [])
        if names:
            print(f"\n   Sample classes (first 5):")
            for i, name in enumerate(names[:5], 1):
                print(f"     {i}. {name}")

            print(f"\n   Sample classes (last 5):")
            for i, name in enumerate(names[-5:], len(names)-4):
                print(f"     {i}. {name}")

        return {
            'exists': True,
            'num_classes': config.get('nc', 0),
            'config': config
        }

    except Exception as e:
        print(f"❌ Error loading YAML: {e}")
        return {'exists': True, 'error': str(e)}


def verify_directory_structure(data_root: Path):
    """Verify data directory structure."""
    print("\n" + "="*60)
    print("📁 VERIFYING DIRECTORY STRUCTURE")
    print("="*60)

    directories = {
        'Elements': data_root / 'Elements',
        'Glyphs': data_root / 'Glyphs'
    }

    results = {}
    total_images = 0
    total_categories = 0

    for name, dir_path in directories.items():
        if dir_path.exists():
            # Count subdirectories (categories)
            categories = [d for d in dir_path.iterdir() if d.is_dir()]
            num_categories = len(categories)

            # Count total images
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            num_images = sum(
                1 for cat_dir in categories
                for img in cat_dir.iterdir()
                if img.suffix.lower() in image_extensions
            )

            results[name] = {
                'exists': True,
                'categories': num_categories,
                'images': num_images
            }

            total_images += num_images
            total_categories += num_categories

            print(f"✅ {name}/")
            print(f"   Categories: {num_categories}")
            print(f"   Images: {num_images:,}")

            # Show sample categories
            print(f"   Sample categories:")
            for cat in sorted(categories, key=lambda x: x.name)[:3]:
                cat_images = len(list(cat.glob('*.jpg'))) + len(list(cat.glob('*.png')))
                print(f"     - {cat.name} ({cat_images} images)")

        else:
            results[name] = {'exists': False}
            print(f"❌ {name}/ - NOT FOUND")

    print(f"\n📈 TOTALS:")
    print(f"   Total categories: {total_categories}")
    print(f"   Total images: {total_images:,}")

    return results


def verify_data_consistency(data_root: Path):
    """Verify consistency between CSV and actual files."""
    print("\n" + "="*60)
    print("🔍 VERIFYING DATA CONSISTENCY")
    print("="*60)

    csv_path = data_root / 'image_data_alljpg.csv'
    if not csv_path.exists():
        print("❌ Cannot verify consistency - CSV file not found")
        return {'verified': False}

    try:
        df = pd.read_csv(csv_path)

        # Check if files in CSV actually exist
        print("Checking if files in CSV exist on disk...")
        sample_size = min(100, len(df))
        sample = df.sample(n=sample_size, random_state=42)

        existing = 0
        missing = 0

        for _, row in sample.iterrows():
            file_path = data_root / row['file_path']
            if file_path.exists():
                existing += 1
            else:
                missing += 1

        print(f"✅ Sample check ({sample_size} files):")
        print(f"   Existing: {existing}/{sample_size}")
        print(f"   Missing: {missing}/{sample_size}")

        # Check statistics
        print(f"\n📊 Dataset Statistics from CSV:")
        print(f"   Total images: {len(df):,}")
        print(f"   Unique categories: {df['full_category'].nunique()}")
        print(f"   Folder distribution:")

        folder_stats = df['folder_type'].value_counts()
        for folder, count in folder_stats.items():
            print(f"     {folder}: {count:,} images")

        # Image size statistics
        print(f"\n📐 Image Size Statistics:")
        print(f"   Average width: {df['width'].mean():.1f}px")
        print(f"   Average height: {df['height'].mean():.1f}px")
        print(f"   Average file size: {df['file_size_kb'].mean():.2f} KB")

        return {
            'verified': True,
            'sample_existing': existing,
            'sample_missing': missing,
            'total_images': len(df)
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {'verified': False, 'error': str(e)}


def main():
    """Main verification function."""
    print("\n" + "="*60)
    print("🔬 CODEX DATASET VERIFICATION")
    print("="*60)

    # Determine data root
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data'

    if data_dir.exists():
        data_root = data_dir
    else:
        data_root = script_dir

    print(f"\n📂 Data root: {data_root.absolute()}")

    # Run all verifications
    csv_results = verify_csv_files(data_root)
    yaml_results = verify_yaml_config(data_root)
    dir_results = verify_directory_structure(data_root)
    consistency_results = verify_data_consistency(data_root)

    # Summary
    print("\n" + "="*60)
    print("📋 VERIFICATION SUMMARY")
    print("="*60)

    all_passed = True

    # Check CSV files
    csv_passed = all(r.get('exists', False) and 'error' not in r for r in csv_results.values())
    print(f"{'✅' if csv_passed else '❌'} CSV Files: {'PASS' if csv_passed else 'FAIL'}")
    all_passed &= csv_passed

    # Check YAML
    yaml_passed = yaml_results.get('exists', False) and 'error' not in yaml_results
    print(f"{'✅' if yaml_passed else '❌'} YAML Config: {'PASS' if yaml_passed else 'FAIL'}")
    all_passed &= yaml_passed

    # Check directories
    dir_passed = all(r.get('exists', False) for r in dir_results.values())
    print(f"{'✅' if dir_passed else '❌'} Directory Structure: {'PASS' if dir_passed else 'FAIL'}")
    all_passed &= dir_passed

    # Check consistency
    consistency_passed = consistency_results.get('verified', False)
    print(f"{'✅' if consistency_passed else '❌'} Data Consistency: {'PASS' if consistency_passed else 'FAIL'}")
    all_passed &= consistency_passed

    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL VERIFICATIONS PASSED!")
        print("✅ Your dataset is ready for YOLO training!")
    else:
        print("⚠️  SOME VERIFICATIONS FAILED")
        print("Please check the errors above and re-run setup scripts if needed.")
    print("="*60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
