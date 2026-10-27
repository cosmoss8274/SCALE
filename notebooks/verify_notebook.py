"""Execute a notebook in a real Jupyter kernel, as far as the machine allows.

    python notebooks/verify_notebook.py                     # local notebook, CPU-only part
    python notebooks/verify_notebook.py --all               # everything, needs a CUDA GPU
    python notebooks/verify_notebook.py SCALE_demo_colab.ipynb --stop-after 3

Without a GPU the oracle cells cannot run, so by default this stops before them and
reports how far it got. That still exercises the imports, the pool, the generator
registry, the objective and the plotting setup in a kernel -- which is the part a plain
``python`` run of the cells does not cover, since it never renders anything.

Exit status is 0 only if every cell attempted ran without raising.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

HERE = Path(__file__).resolve().parent

#: The first code cell that needs a structure predictor on a GPU. Cells before it run
#: anywhere; cells from it onwards need CUDA and an installed Boltz.
FIRST_GPU_CELL = 5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebook", nargs="?", default="SCALE_demo.ipynb")
    parser.add_argument("--all", action="store_true",
                        help="run every cell, including the ones needing a GPU")
    parser.add_argument("--stop-after", type=int, default=None,
                        help="run only this many code cells")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    path = Path(args.notebook)
    if not path.exists():
        path = HERE / args.notebook
    if not path.exists():
        print("no such notebook: %s" % args.notebook, file=sys.stderr)
        return 2

    limit = args.stop_after if args.stop_after is not None else (
        10 ** 9 if args.all else FIRST_GPU_CELL)

    nb = nbformat.read(path, as_version=4)
    code_cells = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
    keep = set(code_cells[:limit])
    skipped = len(code_cells) - len(keep)
    nb.cells = [c for i, c in enumerate(nb.cells) if c.cell_type != "code" or i in keep]

    print("%s: running %d of %d code cells%s"
          % (path.name, len(keep), len(code_cells),
             " (skipping %d that need a GPU; pass --all to include them)" % skipped
             if skipped else ""), flush=True)

    client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}},
                            allow_errors=False)
    try:
        client.execute()
    except CellExecutionError as exc:
        print("\nFAILED:\n%s" % str(exc)[-3000:], file=sys.stderr)
        return 1

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            text = out.get("text") or out.get("data", {}).get("text/plain", "")
            for line in str(text).rstrip().splitlines():
                print("  | %s" % line[:110])

    print("\nOK: %d code cells ran in a kernel" % len(keep))
    if skipped:
        print("    %d GPU cells not attempted on this machine" % skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
