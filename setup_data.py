#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data setup script for Codex project.
Downloads and extracts the image dataset on first run.
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
from typing import Optional
import hashlib

# Configure UTF-8 output for Windows compatibility
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Configuration
DATA_URL = "https://drive.google.com/uc?export=download&id=1oDjC6WKCnzaLWGvXiyCoy5kgf7QE6DR1"  # Replace with your actual URL
DATA_DIR = Path(__file__).parent / "data"
ELEMENTS_DIR = DATA_DIR / "Elements"
GLYPHS_DIR = DATA_DIR / "Glyphs" 
ZIP_FILE = DATA_DIR / "codex-dataset.zip"

# Expected hash of the dataset (for integrity checking)
EXPECTED_HASH = "16071e38fa186327c17d48caed747c40a23c9e8d883d4d74af821f7d88417504"  # Replace with actual hash

def calculate_file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def download_with_progress(url: str, filepath: Path) -> bool:
    """Download file with progress bar, handling Google Drive large file confirmation."""
    try:
        print(f"Downloading dataset from {url}...")

        # Initial request to check if we need confirmation
        session = requests.Session()
        response = session.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()

        # Check if this is the Google Drive virus scan warning page
        content_type = response.headers.get('content-type', '')

        # If we get HTML, we need to extract the confirmation token
        if 'text/html' in content_type:
            import re
            # Get the full HTML content (it's small)
            html_content = b''.join(response.iter_content(chunk_size=8192)).decode('utf-8')

            # Extract the confirmation token and UUID
            confirm_match = re.search(r'name="confirm"\s+value="([^"]+)"', html_content)
            uuid_match = re.search(r'name="uuid"\s+value="([^"]+)"', html_content)
            id_match = re.search(r'name="id"\s+value="([^"]+)"', html_content)

            if confirm_match and id_match:
                confirm = confirm_match.group(1)
                file_id = id_match.group(1)
                uuid = uuid_match.group(1) if uuid_match else None

                # Construct the direct download URL with confirmation
                download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm={confirm}"
                if uuid:
                    download_url += f"&uuid={uuid}"

                print("Large file detected, confirming download...")

                # Make a new request with the confirmation
                response = session.get(download_url, stream=True, allow_redirects=True)
                response.raise_for_status()
            else:
                print("Error: Could not extract confirmation token from Google Drive page")
                return False

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownload progress: {percent:.1f}%", end="", flush=True)

        print("\nDownload completed!")

        # Verify the file is not HTML
        with open(filepath, 'rb') as f:
            first_bytes = f.read(100)
            if first_bytes.startswith(b'<!DOCTYPE html') or first_bytes.startswith(b'<html'):
                print("Error: Downloaded file appears to be HTML, not a zip file")
                return False

        return True
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def extract_dataset(zip_path: Path, extract_to: Path) -> bool:
    """Extract the downloaded dataset."""
    try:
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Dataset extracted successfully!")
        return True
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        return False

def verify_dataset() -> bool:
    """Verify that the dataset was extracted correctly."""
    required_dirs = [ELEMENTS_DIR, GLYPHS_DIR]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            return False
        
        # Check if directories contain files
        if not any(dir_path.iterdir()):
            return False
    
    return True

def setup_data() -> bool:
    """Main data setup function."""
    # Check if data already exists
    if verify_dataset():
        print("✓ Dataset already exists and appears complete.")
        return True
    
    print("Dataset not found. Setting up data...")
    
    # Create directories
    DATA_DIR.mkdir(exist_ok=True)
    
    # Download dataset if not exists
    if not ZIP_FILE.exists():
        if not download_with_progress(DATA_URL, ZIP_FILE):
            return False
    
    # Verify file integrity (optional)
    # Skip hash verification if EXPECTED_HASH is a placeholder
    if EXPECTED_HASH and len(EXPECTED_HASH) == 64 and EXPECTED_HASH != "your-dataset-hash-here":
        print("Verifying file integrity...")
        file_hash = calculate_file_hash(ZIP_FILE)
        if file_hash != EXPECTED_HASH:
            print("Warning: File hash doesn't match expected value!")
            print(f"Expected: {EXPECTED_HASH}")
            print(f"Got:      {file_hash}")
            print("Continuing anyway (to skip verification, update EXPECTED_HASH in setup_data.py)...")
    else:
        print("Skipping hash verification (EXPECTED_HASH not set)")
    
    # Extract dataset
    if not extract_dataset(ZIP_FILE, DATA_DIR):
        return False
    
    # Verify extraction
    if not verify_dataset():
        print("Error: Dataset extraction appears incomplete.")
        return False
    
    # Generate CSV metadata files
    print("\n📊 Generating CSV metadata files...")
    try:
        from data_scripts.data_processing import generate_csv_files
        csv_results = generate_csv_files(DATA_DIR)
        if not csv_results.get("success", False):
            print("⚠️ CSV generation had issues, but continuing...")
    except Exception as e:
        print(f"⚠️ Could not generate CSV files: {e}")
        print("You can generate them later by running: python data_processing.py")
    
    # Clean up zip file (optional)
    try:
        ZIP_FILE.unlink()
        print("🧹 Cleaned up temporary files.")
    except Exception:
        pass
    
    print("✅ Dataset setup completed successfully!")
    print(f"📁 Elements directory: {ELEMENTS_DIR}")
    print(f"📁 Glyphs directory: {GLYPHS_DIR}")
    
    # Print some stats
    try:
        elements_count = sum(1 for _ in ELEMENTS_DIR.rglob("*") if _.is_file())
        glyphs_count = sum(1 for _ in GLYPHS_DIR.rglob("*") if _.is_file())
        print(f"📈 Found {elements_count} element files")
        print(f"📈 Found {glyphs_count} glyph files")
    except Exception:
        pass
    
    return True

def main():
    """CLI entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        print("Force mode: Re-downloading dataset...")
        # Remove existing data
        for path in [ELEMENTS_DIR, GLYPHS_DIR, ZIP_FILE]:
            if path.exists():
                if path.is_file():
                    path.unlink()
                else:
                    import shutil
                    shutil.rmtree(path)
    
    if setup_data():
        print("\n🎉 Ready to use the Codex project!")
        print("You can now run:")
        print("  - python main.py")
        print("  - jupyter notebook main.ipynb")
    else:
        print("\n❌ Setup failed. Please check your internet connection and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()