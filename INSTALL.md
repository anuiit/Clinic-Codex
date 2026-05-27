# Installation Guide

Welcome! You don't need to be a programmer to set up the Clinic Codex annotation tool. This guide will help you get everything running on your computer.

## Step 0: Get the Project Files

Before installing, you need to have the Clinic Codex files on your computer.

1. **Download the project**: Go to [https://github.com/anuiit/clinic-codex](https://github.com/anuiit/clinic-codex), click the green **"Code"** button, and select **"Download ZIP"**.
2. **Extract the files**: Right-click the downloaded ZIP file and select "Extract All" (Windows) or double-click it (Mac). Choose a folder you'll remember, like your Desktop.
3. **Open the folder**: Open the extracted `clinic-codex` folder. This is your project folder.

## Step 1: Install Necessary Software

You'll need three tools installed. They are standard pieces of software used by many applications:

1. **Git**: Used to manage the project files. [Download here](https://git-scm.com/downloads).
2. **Python (3.10 or 3.11)**: This is the engine that runs our AI. 
   - **Important**: Please use version **3.10 or 3.11**. Newer versions (like 3.12 or 3.13) are not yet compatible with the AI libraries we use. [Download Python 3.11 here](https://www.python.org/downloads/release/python-3119/).
3. **Node.js (version 22.12.0 or newer)**: This runs the visual part of the tool and matches the app's `frontend/package.json` requirement. [Download here](https://nodejs.org/).

The automated installer downloads large/networked dependencies, including CPU PyTorch wheels, MobileSAM from GitHub, and model files used by the analysis pipeline.

## Step 2: Automated Installation

### For Mac and Linux Users

1. **Open your Terminal**: You can find this in your Applications folder or by searching for "Terminal".
2. **Go to the project folder**: Type `cd ` (with a space) and then drag your `clinic-codex` folder into the Terminal window. Press Enter.
3. **Run the installer**: Type the following and press Enter:
   ```bash
   bash scripts/install.sh
   ```
   *Note: This can take 5–10 minutes to finish. It is setting up all the AI tools for you.*

### For Windows Users

1. **Open PowerShell**: Search for "PowerShell" in your Start menu.
2. **Go to the project folder**: Type `cd ` (with a space) and then drag your `clinic-codex` folder into the PowerShell window. Press Enter.
3. **Run the installer**: Type the following and press Enter:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
   ```

## Step 3: Final Configuration Check

Before you start the tool, we need to make sure one configuration file is correct.

1. Go into the `frontend` folder inside the project.
2. Look for a file named `.env`.
   - **On Mac**: In Finder, press **Cmd + Shift + .** to show hidden files (files starting with a dot).
   - **On Windows**: In File Explorer, go to View → check "Hidden items".
   - **On Linux**: In your file manager, press **Ctrl + H** to show hidden files.
3. If the `.env` file doesn't exist, create a new text file and name it `.env` (no other extension).
4. Open this file with a text editor (like Notepad or TextEdit).
5. Make sure it contains exactly this line:
   ```
   VITE_API_BASE_URL=http://localhost:7117
   ```
6. Save and close the file.

---

## How to Run the Tool

Whenever you want to use Clinic Codex, follow these steps:

1. Open your Terminal (Mac/Linux) or PowerShell (Windows).
2. Navigate to your project folder using the `cd` command.
3. Start the tool by running:

   **On Mac/Linux**:
   ```bash
   bash scripts/run-dev.sh
   ```

   **On Windows** (in PowerShell):
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\run-dev.ps1
   ```
   Use `scripts/run-dev.ps1` from native Windows PowerShell. On Mac/Linux, use the bash command above rather than `pwsh`.
4. Wait for the message saying the servers have started.
5. Open your web browser (like Chrome or Firefox) and go to:
   `http://localhost:7118`

The normal annotation loop is: upload/analyze, open an analysis for annotation, validate named boxes, send them for local admin review, then open `http://localhost:7118/admin/annotations`. The admin page has Review, Dataset, and Training tabs. Only approved items in the Dataset view are eligible for retraining, and the backend must be restarted after a real retrain.

Advanced users can choose different local ports. On Mac/Linux:
```bash
BACKEND_PORT=7217 FRONTEND_PORT=7218 bash scripts/run-dev.sh
```
On Windows PowerShell:
```powershell
$env:BACKEND_PORT='7217'
$env:FRONTEND_PORT='7218'
powershell -ExecutionPolicy Bypass -File .\scripts\run-dev.ps1
```

Advanced local users can enable the Training tab launch button by setting `ENABLE_ADMIN_TRAINING_JOBS=1` before starting the backend. Leave it disabled unless you are running on your own machine; it is a local convenience wrapper, not a production admin security system.

### ⚠️ A Note on Speed
The first time you analyze an image, the tool will automatically download the AI models (about 40MB). This happens only once.

When you click to analyze a glyph, it usually takes **30–60 seconds** to finish. This is normal because the AI is doing complex math on your computer's processor. Please wait for the result to appear.

## How to Stop

To stop the tool, go back to your Terminal or PowerShell window and press **Ctrl + C** on your keyboard. This will safely shut down the application.

---

## Troubleshooting

- **"Python" or "Node" not found**: Ensure you've installed them from their official websites and restarted your Terminal or PowerShell window.
- **Port already in use**: This usually means the tool is already running in another window. Close that window or stop the process.
- **Analysis fails or never finishes**: 
   - Check that your `frontend/.env` file contains the correct line mentioned in Step 3.
   - Make sure you are using Python 3.10 or 3.11.
- **Backend won't start, error mentions `prototypes.pt`**:
  The model weights need to be exported once before first use. The launcher script (`scripts/run-dev.sh` or `scripts/run-dev.ps1`) does this automatically. If it fails, the source artefact `backend/prototypes/prototypes.pt` may be missing — re-download the project ZIP from GitHub.
- **Optional - Pre-downloading AI models**: If you have a slow internet connection and want to download the AI models before starting, Mac/Linux users can run:
  ```bash
  bash scripts/download-weights.sh
  ```
  Otherwise, the tool will handle this automatically the first time you use it.

---

## Advanced: Manual Step-by-Step Installation

This section is for users who want to see exactly what is happening or need to fix specific issues. You don't need to do this if the automated installation worked.

### 1. Set Up the Python Environment
We use a "virtual environment" to keep the project's tools separate.
```bash
python3.11 -m venv backend/.venv
```

### 2. Install Core Tools
```bash
backend/.venv/bin/pip install --no-cache-dir --prefer-binary numpy pillow pyyaml scipy pandas tqdm
```

### 3. Install Web Framework
```bash
backend/.venv/bin/pip install --no-cache-dir --prefer-binary flask flask-cors
```

### 4. Install AI Engine
```bash
backend/.venv/bin/pip install --no-cache-dir --prefer-binary --extra-index-url https://download.pytorch.org/whl/cpu "torch>=2.1,<2.6" "torchvision>=0.16,<0.21"
```

### 5. Install Image Analysis Tools
```bash
backend/.venv/bin/pip install --no-cache-dir --prefer-binary "segment-anything==1.0" "git+https://github.com/ChaoningZhang/MobileSAM.git" "albumentations>=1.4,<2.0" "timm>=0.9"
```

### 6. Set Up the Web Interface
```bash
cd frontend && npm install && cd ..
```

### 7. Start the Backend Server
```bash
PORT=7117 backend/.venv/bin/python -m flask --app backend.wsgi run --host 0.0.0.0 --port 7117
```

### 8. Start the Web Interface
In a new window:
```bash
cd frontend && VITE_API_BASE_URL=http://localhost:7117 npm run dev -- --host 127.0.0.1 --port 7118 --strictPort
```
