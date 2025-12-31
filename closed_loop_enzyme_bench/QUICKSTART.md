# Quick Start Guide

## For Colab Users

1. **Upload the project to Colab**:
   - Option A: Upload the entire `closed_loop_enzyme_bench` folder to Google Drive
   - Option B: Clone from GitHub (if hosted)

2. **Open Colab and mount Drive** (if using Option A):
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/closed_loop_enzyme_bench
```

3. **Run notebooks in order**:
   - Start with `colab/01_scaffold_preprocess.ipynb`
   - Each notebook installs dependencies automatically
   - Results are saved in `results/` directory

## For Local Users

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Clone ProteinMPNN** (if not using Colab):
```bash
git clone https://github.com/dauparas/ProteinMPNN.git
```

3. **Run experiments**:
```python
from pathlib import Path
from src.data.scaffolds import load_scaffold
from src.generate.proteinmpnn import run_proteinmpnn, read_fasta_sequences
from src.evaluate.esmfold_eval import evaluate_batch

# Load scaffold
OUT = Path("results")
sc = load_scaffold("1AKL", "A", OUT/"scaffolds")

# Generate sequences with ProteinMPNN
fasta = run_proteinmpnn(
    sc.pdb_path, 
    OUT/"mpnn_single", 
    Path("ProteinMPNN"),
    num_seqs=50, 
    sampling_temp=0.2, 
    seed=42
)
seqs = read_fasta_sequences(fasta)

# Evaluate with ESMFold
fold_res = evaluate_batch(
    seqs[:30], 
    model_id="facebook/esmfold_v1", 
    device="cuda",  # or "cpu"
    out_dir=OUT/"pdb"/"single_shot"
)
```

## Troubleshooting

- **CUDA out of memory**: Reduce `max_n` in `evaluate_batch()` or use `device="cpu"`
- **ProteinMPNN not found**: Ensure ProteinMPNN is cloned and path is correct
- **PDB download fails**: Check internet connection and PDB ID validity
- **Import errors**: Run `pip install -r requirements.txt` again
