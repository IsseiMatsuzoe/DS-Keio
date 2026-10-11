"""Execute each chapter end-to-end in a fresh Jupyter kernel."""
from pathlib import Path
import os
import sys
import nbformat
from nbclient import NotebookClient


def main():
    os.environ.setdefault("MPLBACKEND", "Agg")
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "notebooks").glob("*.ipynb"))
    if len(paths) < 7:
        raise RuntimeError("Expected at least seven notebooks")
    for path in paths:
        print(f"Running {path.name} ...", flush=True)
        nb = nbformat.read(path, as_version=4)
        NotebookClient(
            nb, timeout=240, kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
        ).execute()
        print(f"PASS {path.name}", flush=True)
    print("ALL NOTEBOOKS EXECUTED SUCCESSFULLY", flush=True)


if __name__ == "__main__":
    sys.exit(main())
