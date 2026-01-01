# Official Python Installation (Windows)

If you want to run experiments locally on Windows, install Python from `python.org`.

## Steps

1. Open: `https://www.python.org/downloads/`
2. Download Python 3.12.x (or 3.11.x).
3. Run the installer (`.exe`).
4. IMPORTANT: check **"Add Python to PATH"**
5. Click "Install Now"
6. Close PowerShell and open a new window (or reboot if needed).

Verify:

```powershell
python --version
python -m pip --version
```

## Project setup (recommended via `.venv/`)

```powershell
cd "C:\Users\Mr.Josh\Documents\GitHub\bioinformatics\closed_loop_enzyme_bench"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python check_environment.py
python run_experiment_01.py
```

## If Python is still not found

- Restart PowerShell
- Reboot Windows
- Re-run the installer and confirm "Add Python to PATH" is checked

See `TROUBLESHOOTING.md` for more.
