#!/usr/bin/env python3
"""
Script to prepare the dataset for external hosting.
Creates a zip archive of the image data.
"""

import zipfile
import os
from pathlib import Path
import time

def create_dataset_archive():
    """Create a zip archive of the dataset directories."""
    
    project_root = Path(__file__).parent.parent
    original_elements = project_root / "original_data" / "Elements"
    original_glyphs = project_root / "original_data" / "Glyphs"
    
    print(f"Project root: {project_root}")
    print("🔍 Checking for image directories...")
    print(f" - {original_elements} {'(found)' if original_elements.exists() else '(not found)'}")
    print(f" - {original_glyphs} {'(found)' if original_glyphs.exists() else '(not found)'}")

    # Determine which directories exist
    dirs_to_archive = []
    if original_elements.exists():
        dirs_to_archive.append(("Elements", original_elements))
    if original_glyphs.exists():
        dirs_to_archive.append(("Glyphs", original_glyphs))
    
    if not dirs_to_archive:
        print("❌ No image directories found!")
        print("Expected to find: Elements/ or Glyphs/ or original_data/Elements/ or original_data/Glyphs/")
        return False
    
    archive_path = project_root / "codex-dataset.zip"
    
    print("📦 Creating dataset archive...")
    print(f"Output: {archive_path}")
    
    try:
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            total_files = 0
            total_size = 0
            
            for archive_name, source_dir in dirs_to_archive:
                print(f"Adding {archive_name}/ from {source_dir}")
                
                for file_path in source_dir.rglob("*"):
                    if file_path.is_file():
                        # Calculate relative path for archive
                        relative_path = file_path.relative_to(source_dir)
                        archive_file_path = f"{archive_name}/{relative_path}"
                        
                        zipf.write(file_path, archive_file_path)
                        total_files += 1
                        total_size += file_path.stat().st_size
                        
                        if total_files % 1000 == 0:
                            print(f"  Added {total_files} files...")
        
        archive_size = archive_path.stat().st_size
        compression_ratio = (1 - archive_size / total_size) * 100 if total_size > 0 else 0
        
        print(f"✅ Archive created successfully!")
        print(f"📊 Statistics:")
        print(f"   Files archived: {total_files:,}")
        print(f"   Original size: {total_size / (1024*1024):.1f} MB")
        print(f"   Archive size: {archive_size / (1024*1024):.1f} MB")
        print(f"   Compression: {compression_ratio:.1f}%")
        
        # Generate hash for integrity checking
        import hashlib
        print("🔍 Calculating SHA256 hash...")
        hash_sha256 = hashlib.sha256()
        with open(archive_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        file_hash = hash_sha256.hexdigest()
        print(f"📋 SHA256 hash: {file_hash}")
        
        # Save hash to file
        hash_file = project_root / "codex-dataset.zip.sha256"
        with open(hash_file, 'w') as f:
            f.write(f"{file_hash}  codex-dataset.zip\n")
        
        print(f"\n📝 Next steps:")
        print(f"1. Upload {archive_path.name} to your chosen hosting service")
        print(f"2. Update DATA_URL in setup_data.py with the download link")
        print(f"3. Update EXPECTED_HASH in setup_data.py with: {file_hash}")
        print(f"4. Test the download with: python setup_data.py --force")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating archive: {e}")
        return False

if __name__ == "__main__":
    create_dataset_archive()