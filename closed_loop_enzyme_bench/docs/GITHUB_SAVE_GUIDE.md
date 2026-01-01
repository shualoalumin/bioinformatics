# GitHub Results Saving Guide

This guide explains how to automatically save experiment results to GitHub from Colab.

## Overview

The project includes an automated system to save experiment results (CSV tables and PNG figures) to GitHub after each experiment completes. Large files (PDB structures) are excluded to keep the repository size manageable.

## Setup

### 1. Clone Repository (Not Just Mount)

For automatic saving to work, you need to clone the repository:

```python
!git clone https://github.com/YOUR_USERNAME/bioinformatics.git
%cd bioinformatics/closed_loop_enzyme_bench

# Configure Git (one-time)
!git config user.name "Your Name"
!git config user.email "your.email@example.com"
```

**Note:** If you only mount Google Drive, automatic saving won't work (but you can still save manually).

### 2. Create GitHub Personal Access Token

1. Go to: [GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)](https://github.com/settings/tokens)
2. Click "Generate new token (classic)"
3. Name: `Colab Results`
4. Select scope: `repo` (full control of private repositories)
5. Click "Generate token"
6. **Copy the token** (starts with `ghp_`) - you won't see it again!

### 3. Add Token to Colab Secrets

1. In Colab, click the 🔑 icon in the left sidebar
2. Click "Add a secret"
3. Name: `GITHUB_TOKEN`
4. Value: Paste your token
5. Click "Save"

## Usage

### Automatic Saving

Each experiment notebook automatically saves results after completion:

- `02_single_shot_esmf.ipynb` → saves as `exp02_single_shot`
- `03_closed_loop_esmf.ipynb` → saves as `exp03_closed_loop`
- `04_surrogate_active_learning.ipynb` → saves as `exp04_surrogate_guided`
- `05_figures_tables.ipynb` → saves as `exp05_figures_tables`

No action needed - results are automatically committed and pushed!

### Manual Saving

If automatic saving fails or you want to save manually:

```python
from src.utils.github_save import save_results_to_github

# Save with custom experiment name
save_results_to_github("exp02_single_shot")

# Save to different branch
save_results_to_github("exp02_single_shot", branch="experiments")
```

## What Gets Saved

### Included (Automatically Committed)
- ✅ `results/tables/*.csv` - Experimental data tables
- ✅ `results/figures/*.png` - Visualization plots

### Excluded (Not Committed)
- ❌ `results/pdb/*.pdb` - Predicted protein structures (too large)
- ❌ `results/scaffolds/*.pdb` - Original PDB files (can be re-downloaded)
- ❌ `results/mpnn_*/*` - Intermediate ProteinMPNN outputs

This selective approach keeps the repository small (~50KB per experiment) while preserving all important results.

## Commit Messages

Results are saved with descriptive commit messages:

```
Results: exp02_single_shot - 2025-01-15 14:30
Results: exp03_closed_loop - 2025-01-15 16:45
```

Each commit includes a timestamp for easy tracking.

## Troubleshooting

### "Git repository not found"

**Problem:** Repository wasn't cloned, only mounted from Drive.

**Solution:** Clone the repository instead:
```python
!git clone https://github.com/YOUR_USERNAME/bioinformatics.git
%cd bioinformatics/closed_loop_enzyme_bench
```

### "GitHub token not found"

**Problem:** Token not set in Colab Secrets.

**Solution:** 
1. Click 🔑 icon → Add secret
2. Name: `GITHUB_TOKEN`
3. Value: Your token (from GitHub)

### "No results to commit"

**Problem:** No CSV or PNG files found in `results/`.

**Solution:** 
- Check that experiment completed successfully
- Verify files exist: `!ls results/tables/ results/figures/`

### "Push failed"

**Problem:** Permission denied or network issue.

**Solution:**
- Verify token has `repo` scope
- Check internet connection
- Try again (may be temporary network issue)

### "Nothing to commit"

**Problem:** Files already committed (no changes).

**Solution:** This is normal - results are already saved. No action needed.

## Security Notes

- ✅ Token is stored securely in Colab Secrets (encrypted)
- ✅ Token never appears in notebook code
- ✅ Only results files are committed (no sensitive data)
- ⚠️ Don't share notebooks with token hardcoded
- ⚠️ Rotate token if accidentally exposed

## Advanced Usage

### Custom Repository URL

```python
save_results_to_github(
    "exp02_single_shot",
    repo_url="https://github.com/username/repo.git"
)
```

### Environment Variable (Alternative to Colab Secrets)

```python
import os
os.environ['GITHUB_TOKEN'] = 'ghp_xxxxxxxxxxxx'
save_results_to_github("exp02_single_shot")
```

**Note:** This is less secure than Colab Secrets - use only for testing.

## Best Practices

1. **Always clone repository** (don't just mount Drive) for automatic saving
2. **Use Colab Secrets** for token storage (never hardcode)
3. **Review commits** on GitHub to verify results are saved correctly
4. **Keep repository clean** - large files are automatically excluded
5. **Use descriptive experiment names** for easy tracking

## Example Workflow

```python
# 1. Clone repository
!git clone https://github.com/YOUR_USERNAME/bioinformatics.git
%cd bioinformatics/closed_loop_enzyme_bench

# 2. Set Git config (one-time)
!git config user.name "Your Name"
!git config user.email "your.email@example.com"

# 3. Run experiment (notebook automatically saves at end)
# ... run cells in 02_single_shot_esmf.ipynb ...

# 4. Verify on GitHub
# Check: https://github.com/YOUR_USERNAME/bioinformatics/commits/main
```

## Support

If you encounter issues:
1. Check error messages in notebook output
2. Verify setup steps above
3. Check GitHub repository for existing commits
4. Review `.gitignore` to ensure files aren't excluded
