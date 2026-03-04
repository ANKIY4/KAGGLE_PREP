"""Helper module for creating Jupyter notebooks programmatically."""
import nbformat as nbf
import os

OUTPUT_DIR = os.path.expanduser("~/Desktop/Prepare_kaggle")

def new_notebook():
    """Create a new notebook with Python 3 kernel."""
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }
    nb.metadata.language_info = {
        "name": "python",
        "version": "3.10.0"
    }
    return nb

def md(nb, source):
    """Add a markdown cell."""
    nb.cells.append(nbf.v4.new_markdown_cell(source.strip()))

def code(nb, source):
    """Add a code cell."""
    nb.cells.append(nbf.v4.new_code_cell(source.strip()))

def save(nb, filename):
    """Save notebook to output directory."""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w') as f:
        nbf.write(nb, f)
    print(f"Created: {path}")
