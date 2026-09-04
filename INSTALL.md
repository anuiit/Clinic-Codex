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
3. **Node.js (version 22.22.0 or newer)**: This runs the visual part of the tool and matches the app's `frontend/package.json` requirement. [Download here](https://nodejs.org/).

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
   Keep the project on a local Windows drive, for example `C:\Projects\Clinic Codex`.
   Paths with spaces are supported. UNC/network paths and `\\wsl.localhost\...`
   are not supported by the native launchers; use the Bash scripts inside WSL instead.
2. **Go to the project folder**: Type `cd ` (with a space) and then drag your `clinic-codex` folder into the PowerShell window. Press Enter.
3. **Run the installer** using the command matching the terminal you opened:

   Windows PowerShell 5.1:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
   ```

   PowerShell 7 on Windows:
   ```powershell
   pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\install.ps1
   ```

## Step 3: Final Configuration Check

There is nothing to configure by hand. The installer creates `backend/.env` with authentication enabled, a unique random session secret, and the loopback HTTP cookie setting. Re-running it preserves every existing value and only adds missing settings. It never creates a default email or password.

Do not create `frontend/.env` for the standard local setup. The launchers configure the frontend API address automatically.

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

   **On Windows PowerShell 5.1**:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\run-dev.ps1
   ```

   **On PowerShell 7 for Windows**:
   ```powershell
   pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-dev.ps1
   ```

   Use the `.ps1` scripts only on native Windows. On Mac/Linux/WSL, use the Bash command above.
4. Wait for the message saying the servers have started.
5. Open your web browser (like Chrome or Firefox) and go to:
   `http://localhost:7118`

### Create the first local administrator

On a clean installation, the application redirects you to `http://localhost:7118/login`. Because no account exists yet, this page offers to create the first local administrator:

1. Enter the email and password that you want to use locally.
2. Create the administrator account. This option is available only once.
3. Sign in with the same email and password.

The account receives access to the administration area. No password is shipped in the project and none is printed by the installer.

If support asks you to check the installation without leaving the application running, use:

- Mac/Linux/WSL: `bash scripts/run-dev.sh --smoke`
- Windows PowerShell 5.1: `powershell -ExecutionPolicy Bypass -File .\scripts\run-dev.ps1 -Smoke`
- PowerShell 7 on Windows: `pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-dev.ps1 -Smoke`

The smoke starts both services, verifies model-file integrity through `/ready`, prints `smoke PASS`, and shuts them down. This checks startup, not accuracy.

Native Windows validation: Windows 10 x64, Python 3.11.4, Node 24.19.0,
Windows PowerShell 5.1 and PowerShell 7.6.5. Installation, startup, browser annotation,
approval and real CPU candidate creation were exercised from a clone with spaces
in its path. See [the validation report](docs/stable-release-new-user-validation-20260904.md).
Windows GPU/ARM and macOS were not exercised in this release validation.

The standard local mode combines the **shipped model base and all current approved annotations**. It needs no private corpus or manual snapshot. The installer downloads fixed MobileSAM and DINOv2 assets, enables local training, and permits the initial local administrator to review their own annotations.

1. Upload and analyze an image, open its annotation editor, correct boxes and labels, and mark the desired elements ready.
2. Send the annotations, then open **Admin → Review** and approve them.
3. Open **Training**, run the dry run, then select **Non, entraînement complet** and launch.
4. Inspect the candidate path and result in Training.

Each run captures current approved crops and review decisions automatically. Exact duplicate images count once; conflicting labels for identical images are rejected. Stale decisions and missing crops are excluded. Repeating the same approvals does not count their contribution twice.

The backbone and projection stay frozen. The update adapts prototypes for existing base-model classes; it does not train MobileSAM or introduce new classes. The original base provides the prior even when its training images are unavailable.

Candidates are stored under `backend/model_registry/versions/<version_id>/`, with provenance and checksums. They are **not activated**; promotion is blocked because this local mode has no independent holdout. Reported base/candidate scores measure training-image fit, not better generalization. The running model is unchanged and no restart is needed.


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

The installer enables training and initial-admin self-review in `backend/.env`, preserving existing explicit settings. Both features default to disabled outside the local installation setup.

### ⚠️ A Note on Speed
The installer downloads MobileSAM (about 39 MiB) and DINOv2 (about 84 MiB plus source). Analysis and retraining then use verified local assets.

When you click to analyze a glyph, it usually takes **30–60 seconds** to finish. This is normal because the AI is doing complex math on your computer's processor. Please wait for the result to appear.

## How to Stop

To stop the tool, go back to your Terminal or PowerShell window and press **Ctrl + C** on your keyboard. This will safely shut down the application.

---

## Troubleshooting

- **"Python" or "Node" not found**: Ensure you've installed them from their official websites and restarted your Terminal or PowerShell window.
- **Port already in use**: This usually means the tool is already running in another window. Close that window or stop the process.
- **Analysis fails or never finishes**: Make sure you are using Python 3.10 or 3.11 and review the backend error shown in the launcher terminal.
- **Authentication secret is missing**: Re-run the installer. It adds only missing variables to `backend/.env` and preserves existing settings.
- **Backend won't start, error mentions `prototypes.pt`**:
  The model weights need to be exported once before first use. The launcher script (`scripts/run-dev.sh` or `scripts/run-dev.ps1`) does this automatically. If it fails, the source artefact `backend/prototypes/prototypes.pt` may be missing — re-download the project ZIP from GitHub.
- **Optional - Pre-downloading AI models**: If you have a slow internet connection and want to download the AI models before starting, Mac/Linux users can run:
  ```bash
  bash scripts/download-weights.sh
  ```
  The installer already performs this download. This command also repairs invalid checkpoints. On native Windows, run `backend/.venv/Scripts/python.exe scripts/download_weights.py`.

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

### 7. Complete model and authentication setup

Run `bash scripts/install.sh` to export the base, download assets and create local configuration. Start both services with `bash scripts/run-dev.sh`.

### 8. Start the Backend Server (manual alternative; load backend/.env first)
```bash
PORT=7117 backend/.venv/bin/python -m flask --app backend.wsgi run --host 127.0.0.1 --port 7117
```

### 9. Start the Web Interface
In a new window:
```bash
cd frontend && VITE_API_BASE_URL=http://localhost:7117 npm run dev -- --host 127.0.0.1 --port 7118 --strictPort
```
