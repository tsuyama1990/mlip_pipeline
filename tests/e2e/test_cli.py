import pytest
from typer.testing import CliRunner
from mlip_autopipec.main import app
from pathlib import Path
from ase.io import write
from ase import Atoms
import yaml

runner = CliRunner()

def test_cli_run_success(tmp_path: Path) -> None:
    # Setup config
    config_data = {
        "max_cycles": 1,
        "initial_structure_path": str(tmp_path / "init.xyz"),
        "workdir": str(tmp_path / "work"),
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "generator": {"type": "mock"},
        "validator": {"type": "mock"},
        "selector": {"type": "mock"},
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    # Dummy init
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    write(tmp_path / "init.xyz", atoms)

    result = runner.invoke(app, ["run", str(config_path)])
    assert result.exit_code == 0
    assert "Pipeline finished successfully" in result.stdout

def test_cli_config_not_found(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", str(tmp_path / "nonexistent.yaml")])
    # Typer raises exit code 2 for validation errors (file not found)
    assert result.exit_code == 2
    assert "does not exist" in result.stderr

def test_cli_invalid_config(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_config.yaml"
    with open(config_path, "w") as f:
        f.write("invalid: yaml: content")

    result = runner.invoke(app, ["run", str(config_path)])
    # Pydantic validation error or just missing fields
    assert result.exit_code == 1
    assert "validation failed" in result.stdout or "error" in result.stdout
