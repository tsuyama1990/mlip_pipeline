from pathlib import Path

import yaml
from typer.testing import CliRunner

from mlip_autopipec.main import app

runner = CliRunner()


def test_init_command(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", "--output", str(tmp_path / "config.yaml")])
    assert result.exit_code == 0
    assert "Created default configuration" in result.output
    assert (tmp_path / "config.yaml").exists()


def test_run_command_valid(tmp_path: Path) -> None:
    # Prepare a valid config
    work_dir = tmp_path / "run_dir"
    config_content = {
        "orchestrator": {"work_dir": str(work_dir), "max_iterations": 2},
        "generator": {"type": "RANDOM", "num_structures": 5},
        "oracle": {"type": "QUANTUM_ESPRESSO", "command": "pw.x"},
        "trainer": {"type": "PACEMAKER", "r_cut": 4.0, "max_deg": 2},
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_content, f)

    result = runner.invoke(app, ["run", str(config_file)])
    assert result.exit_code == 0
    assert "Configuration loaded successfully" in result.output
    assert "Starting new run" in result.output
    assert work_dir.exists()
    assert (work_dir / "workflow_state.json").exists()


def test_run_command_invalid(tmp_path: Path) -> None:
    # Prepare an invalid config (missing required field)
    config_content = {
        "orchestrator": {"work_dir": str(tmp_path / "bad_run")},
        # Missing generator, oracle, trainer
    }
    config_file = tmp_path / "bad_config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_content, f)

    result = runner.invoke(app, ["run", str(config_file)])
    assert result.exit_code == 1
    assert "Configuration Validation Error" in result.output
