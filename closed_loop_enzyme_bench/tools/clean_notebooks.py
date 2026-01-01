from __future__ import annotations

import json
from pathlib import Path


def clean_notebook(nb: dict) -> dict:
    """Remove execution outputs and counts to keep notebooks clean for sharing."""
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return nb


def main() -> int:
    colab_dir = Path(__file__).resolve().parents[1] / "colab"
    paths = sorted(colab_dir.glob("*.ipynb"))
    if not paths:
        print(f"No notebooks found in {colab_dir}")
        return 1

    changed = 0
    for p in paths:
        nb = json.loads(p.read_text(encoding="utf-8"))
        before = json.dumps(nb, sort_keys=True)
        nb = clean_notebook(nb)
        after = json.dumps(nb, sort_keys=True)
        if before != after:
            p.write_text(json.dumps(nb, indent=2), encoding="utf-8")
            changed += 1
    print(f"Cleaned {changed}/{len(paths)} notebooks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

