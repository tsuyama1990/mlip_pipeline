from pathlib import Path

import yaml
from typer.testing import CliRunner

from mlip_autopipec.main import app

runner = CliRunner()

def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.stdout

def test_cli_run_mock(tmp_path: Path) -> None:
    # Create config
    config_data = {
        "work_dir": str(tmp_path),
        "max_cycles": 1,
        "explorer": {"type": "mock"},
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock", "potential_output_name": "cli_potential.yace"},
        "validator": {"type": "mock"}
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    # Change CWD to tmp_path so "cli_potential.yace" is written there?
    # Typer runner doesn't change CWD automatically.
    # MockTrainer writes to CWD.
    # We should run this test carefully.

    result = runner.invoke(app, ["run", "--config", str(config_file)])
    assert result.exit_code == 0
    assert "All cycles finished" in result.stdout

    # Verify artifacts
    dataset_file = tmp_path / "accumulated_dataset.xyz"
    assert dataset_file.exists()

    # Check if potential was created in work_dir
    pot_file = tmp_path / "cli_potential.yace"
    assert pot_file.exists()

def test_cli_run_missing_config() -> None:
    result = runner.invoke(app, ["run", "--config", "nonexistent.yaml"])
    assert result.exit_code == 1
    assert "not found" in result.stderr

def test_cli_run_invalid_config(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.yaml"
    with config_file.open("w") as f:
        f.write("invalid: yaml: content")

    result = runner.invoke(app, ["run", "--config", str(config_file)])
    assert result.exit_code == 1
