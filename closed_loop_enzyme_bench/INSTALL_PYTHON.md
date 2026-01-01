# Installing Python on Windows (for local runs)

This repository works best with a standard Python installation from `python.org`.

## 1) Check whether Python is installed

```powershell
python --version
python -m pip --version
where python
```

## 2) Remove Microsoft Store Python (optional)

If you previously installed Python from the Microsoft Store, it can sometimes cause PATH/conflict issues.

Settings -> Apps -> search "Python" -> Uninstall.

Or (Admin PowerShell):

```powershell
Get-AppxPackage *Python* | Remove-AppxPackage
```

## 3) Install official Python (recommended)

1. Download: `https://www.python.org/downloads/`
2. Run the `.exe` installer.
3. IMPORTANT: check **"Add Python to PATH"**
4. Click "Install Now"
5. Close PowerShell and open a new window (or reboot if needed).

Verify:

```powershell
python --version
python -m pip --version
```

## 4) Install project dependencies (recommended via `.venv/`)

From `closed_loop_enzyme_bench/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python check_environment.py
```

## Troubleshooting

If `python` is not found, verify PATH or reboot. See `TROUBLESHOOTING.md` for more fixes.
