# Troubleshooting (Local runs)

This document lists common issues when running the benchmark locally on Windows.

## 1) `python` is not found

Symptoms:

```
'python' is not recognized...
```

Fix:

- Install Python from `https://www.python.org/downloads/`
- Re-run the installer with **"Add Python to PATH"** checked
- Restart PowerShell (or reboot)

Verify:

```powershell
python --version
where python
```

## 2) Missing dependencies (`torch`, `transformers`, `pandas`, ...)

Recommended fix: use the project virtualenv.

```powershell
cd closed_loop_enzyme_bench
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python check_environment.py
```

## 3) ESMFold is slow / looks stuck

First run downloads ~1GB model weights. On CPU, folding can take minutes per sequence.

Recommendations:
- Start with **1 sequence** on CPU, then scale up
- Use Colab (GPU) for faster iteration

## 4) GPU/CUDA issues

If CUDA is unavailable, the code will fall back to CPU automatically.

Verify:

```python
import torch
print(torch.cuda.is_available())
```

## 5) ProteinMPNN not found

If ProteinMPNN is missing, some scripts will fall back to a mutation baseline.

Install locally:

```powershell
git clone https://github.com/dauparas/ProteinMPNN.git
python -m pip install -r ProteinMPNN/requirements.txt
```

## 6) Windows console encoding errors (UnicodeEncodeError)

If you see encoding errors printing special symbols, update scripts to use ASCII output
or set UTF-8 output in PowerShell:

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

## 7) Still stuck?

Run and share:

```powershell
python check_environment.py
python -c "import sys; print(sys.executable); print(sys.version)"
```
