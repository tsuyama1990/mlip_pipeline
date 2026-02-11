from pathlib import Path

import yaml
from typer.testing import CliRunner

from mlip_autopipec.main import app

runner = CliRunner()


def test_init_command(tmp_path: Path) -> None:
    """Test initializing a new project."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["init", "my_project"])

        assert result.exit_code == 0
        assert "Initialized project in 'my_project'" in result.output

        project_dir = Path("my_project")
        assert project_dir.exists()
        assert (project_dir / "config.yaml").exists()
        assert (project_dir / "data").exists()


def test_init_existing_fails(tmp_path: Path) -> None:
    """Test that initializing an existing directory fails."""
    (tmp_path / "existing").mkdir()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("existing").mkdir(exist_ok=True)

        result = runner.invoke(app, ["init", "existing"])
        assert result.exit_code == 1
        assert "already exists" in result.output


def test_run_loop_command(tmp_path: Path) -> None:
    """Test running the loop via CLI."""
    work_dir = tmp_path / "test_run"

    # Create project manually to control config
    work_dir.mkdir()
    config_path = work_dir / "config.yaml"

    config_data = {
        "orchestrator": {
            "work_dir": str(work_dir),
            "n_iterations": 1,
        },
        "generator": {"type": "mock", "n_candidates": 5},
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "validator": {"type": "mock"},
    }

    with config_path.open("w") as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["run-loop", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "Pipeline completed successfully" in result.output

    assert (work_dir / "iter_001" / "train.xyz").exists()
    assert (work_dir / "workflow_state.json").exists()


def test_run_loop_invalid_config(tmp_path: Path) -> None:
    """Test CLI handles invalid config gracefully."""
    config_path = tmp_path / "bad_config.yaml"

    # Missing required fields
    config_data = {
        "orchestrator": {"n_iterations": -1} # Invalid
    }

    with config_path.open("w") as f:
        yaml.dump(config_data, f)

    result = runner.invoke(app, ["run-loop", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "Configuration Error" in result.output
