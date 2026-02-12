from pathlib import Path

import yaml
from typer.testing import CliRunner

from mlip_autopipec.main import app

runner = CliRunner()


def test_cli_init(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", "--output", str(tmp_path / "config.yaml")])
    assert result.exit_code == 0
    assert (tmp_path / "config.yaml").exists()

    with (tmp_path / "config.yaml").open() as f:
        data = yaml.safe_load(f)
    assert "orchestrator" in data


def test_cli_run_success(tmp_path: Path) -> None:
    # First init
    config_file = tmp_path / "config.yaml"
    work_dir = tmp_path / "run_work"

    config_data = {
        "orchestrator": {"work_dir": str(work_dir)},
        "generator": {"type": "RANDOM"},
        "oracle": {"type": "QUANTUM_ESPRESSO", "command": "pw.x"},
        "trainer": {"type": "PACEMAKER"},
    }

    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["run", str(config_file)])
    assert result.exit_code == 0
    assert "Starting new run" in result.output
    assert (work_dir / "workflow_state.json").exists()

    # Resume
    result = runner.invoke(app, ["run", str(config_file)])
    assert result.exit_code == 0
    assert "Resuming from iteration 0" in result.output


def test_cli_run_invalid_config(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.yaml"
    with config_file.open("w") as f:
        f.write("invalid: yaml")

    result = runner.invoke(app, ["run", str(config_file)])
    assert result.exit_code == 1
    assert "Configuration Validation Error" in result.output
