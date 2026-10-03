"""Check that the demo notebook is valid and its code cells compile."""
from pathlib import Path

import nbformat


def test_notebook_code_cells_compile():
    path = Path(__file__).parents[1] / "notebooks" / "AutoMage_Demo.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"{path.name}:cell{index}", "exec")
