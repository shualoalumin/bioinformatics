#!/usr/bin/env python3
"""
Check if the environment is ready for experiments.
"""
import sys
from pathlib import Path

def check_python():
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    return True

def check_packages():
    """Check if required packages are installed."""
    required = {
        'torch': 'PyTorch',
        'transformers': 'Transformers (Hugging Face)',
        'biopython': 'BioPython',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn',
        'yaml': 'PyYAML',
    }
    
    missing = []
    installed = []
    
    for module, name in required.items():
        try:
            if module == 'yaml':
                __import__('yaml')
            elif module == 'sklearn':
                __import__('sklearn')
            else:
                __import__(module)
            installed.append(name)
            print(f"✓ {name}")
        except ImportError:
            missing.append(name)
            print(f"✗ {name} - MISSING")
    
    return missing, installed

def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("⚠ CUDA not available - will use CPU (slower)")
        return cuda_available
    except ImportError:
        print("✗ PyTorch not installed - cannot check CUDA")
        return False

def main():
    print("=" * 60)
    print("Environment Check for Closed-loop Enzyme Benchmark")
    print("=" * 60)
    
    print("\n1. Python:")
    check_python()
    
    print("\n2. Required Packages:")
    missing, installed = check_packages()
    
    print("\n3. CUDA/GPU:")
    cuda_available = check_cuda()
    
    print("\n" + "=" * 60)
    if missing:
        print("⚠ Some packages are missing!")
        print("\nInstall missing packages with:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("✓ All required packages are installed!")
        if not cuda_available:
            print("\n⚠ Note: Running on CPU will be slower.")
            print("  Consider using Google Colab for GPU acceleration.")
        return True

if __name__ == "__main__":
    ready = main()
    sys.exit(0 if ready else 1)
