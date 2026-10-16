# Run Noor in VS Code or another IDE

Python runs the application server; the interface opens in your browser. VS Code is the editor and terminal. You do not need a GPU or to train again when using the complete prototype ZIP.

## Start with the complete prototype

Download and extract `Noor-Persian-Letter-Lab.zip` from the project conversation. Open the extracted `Noor-Persian-Letter-Lab` folder in VS Code using **File → Open Folder**. Choose the folder containing `README.md`, `requirements.lock`, `persian_ocr`, `data`, and `artifacts`, not its parent folder.

The GitHub branch contains the source, but excludes datasets and trained model binaries. To use a GitHub checkout, copy `data/` and `artifacts/` from the complete ZIP into its project root. Otherwise the interface opens with recognition unavailable until a model is trained.

Install Python **3.12, 64-bit** (x64 on Windows), VS Code, and Microsoft's **Python** extension. Python 3.15 is unsupported by the pinned PyTorch wheels; changing the project's version restriction will not make those wheels compatible. The project recommends the Python Debugger extension as well. Initial dependency installation needs Internet access and sufficient free space for PyTorch and the other packages. Inference runs on the CPU.

## Windows: VS Code PowerShell terminal

Open **Terminal → New Terminal** in the project folder. Run these commands one at a time:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
.\.venv\Scripts\python.exe -m uvicorn persian_ocr.api:app --host 127.0.0.1 --port 8000
```

The setup script looks for Python 3.12 x64 in the usual installation folders, through `py -3.12`, and on `PATH`. It creates `.venv`, verifies the actual environment's version and architecture, installs the hashed lock, and runs `pip check`. It does not download or install Python. `-ExecutionPolicy Bypass` applies only to this PowerShell process; it does not change the saved execution policy. Activation is unnecessary.

If Python is installed elsewhere, pass its executable explicitly (replace the example path with your installation):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -PythonExe "C:\path\to\Python312\python.exe"
```

If `.venv` already uses another Python version, setup stops without modifying it. Close terminals and running processes that use that environment, preserve it with the command below, and rerun setup. Choose a different backup name if this one already exists. Do not create a new environment over an incompatible existing one.

```powershell
Rename-Item -LiteralPath .venv -NewName .venv-incompatible-backup
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

On **your computer**, open `http://127.0.0.1:8000` in a browser. Leave the terminal running while using the app. Press **Ctrl+C** in that terminal to stop it. On subsequent runs you only need the `uvicorn` command; dependency installation is a one-time step unless the lock changes.

## Run using F5

After installing dependencies, open the Command Palette and select **Python: Select Interpreter**. Choose `.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on Linux. In **Run and Debug**, choose **Noor: Run web app**, then press **F5**. Open the same local browser address. Stop an existing terminal server before starting another on port 8000.

## Linux

From the project root:

```bash
bash scripts/setup.sh
source .venv/bin/activate
bash scripts/run.sh
```

The prepared cloud instance already has its environment. Its startup commands are:

```bash
cd /workspace/BigDataAnalysis
source /workspace/.cloud-setup/bigdataanalysis/env.sh
bash scripts/run.sh
```

The cloud address is internal to that machine. Opening the cloud interface from another device requires the host's port-forwarding or hosting support. Opening a local address on your computer does not connect to the cloud machine.

## macOS

The current shared lock uses Linux/Windows `+cpu` PyTorch builds. Do not run that lock unchanged on macOS. Apple Silicon and Intel Macs need platform-compatible PyTorch wheels and a corresponding dependency lock. The prepared cloud copy is an alternative while a Mac-specific environment is configured.

## First check

1. The page should show **MobileNetV2 ready**.
2. Upload `data/letters/images/validation/DejaVuSans_00_00.png` and click **Recognize character**. The provided synthetic example is `ا` (Alef).
3. Open **Dataset studio** and check that it lists 100 pages.
4. Open **Evaluation** to see the measured results. Running the app does not change the model's reported accuracy.

If Python reports `No module named uvicorn`, use the `.venv` interpreter shown above. If the app reports an unavailable model, check that `artifacts/model_metadata.json` and `artifacts/mobilenet.pt` were extracted into the project root. If port 8000 is occupied, stop the previous server or change the command's port to 8001 and use that port in the browser.

PyCharm and other IDEs use the same environment and module command. Select the `.venv` interpreter and run `python -m uvicorn persian_ocr.api:app --host 127.0.0.1 --port 8000` with the project root as the working directory.
