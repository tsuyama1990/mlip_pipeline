import os
import subprocess
import sys
import pytest
from pathlib import Path

# Locate tutorials directory relative to this test file
TUTORIALS_DIR = Path(__file__).parent.parent.parent / "tutorials"
NOTEBOOKS = sorted([nb for nb in TUTORIALS_DIR.glob("*.ipynb")])

@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_execution(notebook_path: Path):
    """
    Executes a Jupyter notebook and asserts that it finishes successfully.
    Sets CI=true to trigger Mock mode in the notebook.
    """
    if not notebook_path.exists():
        pytest.fail(f"Notebook {notebook_path} does not exist")

    print(f"Executing notebook: {notebook_path}")

    env = os.environ.copy()
    env["CI"] = "true"

    # Use sys.executable to ensure we use the same python environment
    cmd = [
        sys.executable,
        "-m", "jupyter",
        "nbconvert",
        "--to", "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=600",
        "--stdout",
        str(notebook_path)
    ]

    # Run the command
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print("Notebook execution failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        pytest.fail(f"Notebook {notebook_path.name} failed to execute. Return code: {result.returncode}\nError: {result.stderr}")

    print(f"Notebook {notebook_path.name} passed successfully.")
