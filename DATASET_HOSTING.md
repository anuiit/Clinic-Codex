# Codex Dataset Hosting Options

This document outlines options for hosting the dataset externally.

## Option 1: Google Drive (Recommended for testing)

1. Upload your dataset as a zip file to Google Drive
2. Make it publicly shareable
3. Get the direct download link:
   - Right-click file → Share → Anyone with the link
   - Copy the file ID from the URL: `https://drive.google.com/file/d/FILE_ID/view`
   - Use this format: `https://drive.google.com/uc?export=download&id=FILE_ID`

## Option 2: GitHub Releases (For smaller datasets)

If you can compress the dataset to <2GB:
1. Create a release on GitHub
2. Attach the dataset zip as a release asset
3. Use the direct download URL from the release

## Option 3: AWS S3 / Google Cloud Storage (Production)

For production use:
1. Upload to cloud storage
2. Configure public read access
3. Use the direct HTTP URL

## Option 4: Academic/Research Storage

Many institutions provide storage for research data:
- Zenodo (up to 50GB per dataset)
- Figshare
- University repositories

## Implementation

Once you choose a hosting option, update the `DATA_URL` in `setup_data.py`:

```python
DATA_URL = "https://your-chosen-storage-url.com/codex-dataset.zip"
```

## Creating the Dataset Archive

To prepare your dataset for upload:

```bash
# Navigate to your project directory
cd C:\Users\anmou\Desktop\Codex

# Create the dataset archive
# Option 1: Include both Elements and Glyphs
7z a -tzip codex-dataset.zip Elements/ Glyphs/

# Option 2: Include original_data structure
7z a -tzip codex-dataset.zip original_data/

# Check the archive size
ls -lh codex-dataset.zip
```

The archive should maintain the directory structure so it extracts correctly.