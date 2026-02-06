from pathlib import Path

import yaml
from typer.testing import CliRunner

from mlip_autopipec.main import app

runner = CliRunner()


def test_cli_help() -> None:
    """
    Tests that the CLI help command works.
    """
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.stdout


def test_cli_run_mock(tmp_path: Path) -> None:
    """
    Tests running the pipeline via CLI with a mock configuration.
    """
    # Create config
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    config_data = {
        "work_dir": str(work_dir),
        "max_cycles": 1,
        "random_seed": 42,
        "explorer": {"type": "mock", "n_structures": 2},
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock", "potential_output_name": "cli_potential.yace"},
        "validator": {"type": "mock"},
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["run", "--config", str(config_file)])

    assert result.exit_code == 0
    assert "All cycles finished" in result.stdout

    # Verify artifacts
    dataset_file = work_dir / "accumulated_dataset.xyz"
    assert dataset_file.exists()

    # Check if potential was created in work_dir
    pot_file = work_dir / "cli_potential.yace"
    assert pot_file.exists()


def test_cli_run_missing_config() -> None:
    """
    Tests that running with a missing config file returns an error.
    """
    result = runner.invoke(app, ["run", "--config", "nonexistent.yaml"])
    assert result.exit_code == 1
    assert "not found" in result.stderr


def test_cli_run_invalid_config(tmp_path: Path) -> None:
    """
    Tests that running with an invalid config file returns an error.
    """
    config_file = tmp_path / "bad.yaml"
    with config_file.open("w") as f:
        f.write("invalid: yaml: content")

    result = runner.invoke(app, ["run", "--config", str(config_file)])
    assert result.exit_code == 1
