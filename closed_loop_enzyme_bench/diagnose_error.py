import sys
import traceback
from pathlib import Path

def log(msg):
    with open("debug_log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)

def main():
    log("=== Starting Diagnostic Run ===")
    log(f"Python: {sys.version}")
    log(f"Executable: {sys.executable}")
    
    try:
        log("Checking imports...")
        import torch
        log(f"PyTorch version: {torch.__version__}")
        import Bio
        log(f"BioPython version: {Bio.__version__}")
        import transformers
        log(f"Transformers version: {transformers.__version__}")
        
        log("Running load_scaffold...")
        from src.data.scaffolds import load_scaffold
        
        OUT = Path("results")
        log(f"Output directory: {OUT.absolute()}")
        
        sc = load_scaffold("1AKL", "A", OUT / "scaffolds")
        log(f"Success! Scaffold length: {len(sc.sequence)}")
        
    except Exception as e:
        log(f"\nERROR OCCURRED:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
