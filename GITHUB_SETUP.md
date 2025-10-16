# GitHub Setup Guide for Codex Project

## Steps to Make Your Project GitHub-Ready

### 1. Prepare the Dataset Archive

First, create a compressed archive of your image data:

```bash
python create_dataset_archive.py
```

This will create `codex-dataset.zip` with your Elements and Glyphs directories.

### 2. Choose a Hosting Option

**Option A: Google Drive (Easiest for testing)**
1. Upload `codex-dataset.zip` to Google Drive
2. Right-click → Share → "Anyone with the link can view"
3. Copy the file ID from the URL
4. Format: `https://drive.google.com/uc?export=download&id=FILE_ID`

**Option B: GitHub Releases (If <2GB)**
1. Create a release on GitHub
2. Attach `codex-dataset.zip` as a release asset
3. Use the direct download URL

**Option C: Cloud Storage (Production)**
- AWS S3, Google Cloud Storage, etc.
- Configure public read access

### 3. Update Configuration

Edit `setup_data.py` and update these lines:

```python
DATA_URL = "https://your-actual-download-url.com/codex-dataset.zip"
EXPECTED_HASH = "paste-the-hash-from-create_dataset_archive.py-output"
```

### 4. Test the Setup

```bash
# Remove existing data directories
rm -rf Elements Glyphs original_data/Elements original_data/Glyphs

# Test the download
python setup_data.py --force
```

### 5. Initialize Git Repository (if not done)

```bash
git init
git add .
git commit -m "Initial commit: Codex glyph detection project"
```

### 6. Create GitHub Repository

1. Go to GitHub and create a new repository named "codex"
2. Don't initialize with README (you already have one)

```bash
git remote add origin https://github.com/yourusername/codex.git
git branch -M main
git push -u origin main
```

### 7. Update Repository URLs

Edit `pyproject.toml` and replace `yourusername` with your actual GitHub username.

## Final Repository Structure

Your GitHub repo will contain:

```
codex/
├── .github/workflows/ci.yml    # GitHub Actions
├── .gitignore                  # Excludes large files
├── README.md                   # Project documentation
├── pyproject.toml             # Dependencies & metadata
├── setup_data.py              # Data download script
├── create_dataset_archive.py   # Utility for creating archives
├── main.py                    # Main script
├── main.ipynb                 # Jupyter notebook
├── data_scripts/              # Data processing utilities
├── models/                    # Model files (smaller ones)
├── synthetic_dataset/         # Generated data
└── runs/                      # Training outputs (gitignored)
```

## User Experience

When someone wants to use your project:

```bash
# 1. Clone the repo (fast - no large files)
git clone https://github.com/yourusername/codex.git
cd codex

# 2. Install dependencies
pip install -e .

# 3. Download dataset (automated)
python setup_data.py

# 4. Start using
python main.py
```

## Benefits of This Approach

✅ **Fast cloning** - No large files in Git  
✅ **Clean repository** - Focus on code  
✅ **Automated setup** - One command to get data  
✅ **Flexible hosting** - Easy to change data source  
✅ **Integrity checking** - Hash verification  
✅ **Scalable** - Easy to add more datasets  

## Troubleshooting

**If users can't download the data:**
- Check the DATA_URL is accessible
- Verify file permissions on hosting service
- Test with `python setup_data.py --force`

**If the hash doesn't match:**
- Re-create the archive with `create_dataset_archive.py`
- Update the EXPECTED_HASH in `setup_data.py`