from pathlib import Path

import yaml
from typer.testing import CliRunner

from mlip_autopipec.main import app

runner = CliRunner()


def test_init_command(tmp_path: Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["init", "--output", "config.yaml"])
        assert result.exit_code == 0, result.stdout
        assert Path("config.yaml").exists(), f"File missing in {Path.cwd()}. Output: {result.stdout}"

        with Path("config.yaml").open() as f:
            data = yaml.safe_load(f)
        assert data["oracle"]["type"] == "mock"


def test_check_config_valid(tmp_path: Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Create valid config
        config = {
            "workdir": "tmp",
            "oracle": {"type": "mock"},
            "trainer": {"type": "mock"},
            "dynamics": {"type": "mock"},
            "generator": {"type": "mock"},
            "validator": {"type": "mock"},
            "selector": {"type": "mock"},
        }
        with Path("config.yaml").open("w") as f:
            yaml.dump(config, f)

        result = runner.invoke(app, ["check-config", "config.yaml"])
        assert result.exit_code == 0
        assert "Configuration is valid" in result.stdout


def test_check_config_invalid(tmp_path: Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Create invalid config
        config = {
            "workdir": "tmp",
            "max_cycles": 0,  # Invalid
            "oracle": {"type": "mock"},
            "trainer": {"type": "mock"},
            "dynamics": {"type": "mock"},
            "generator": {"type": "mock"},
            "validator": {"type": "mock"},
            "selector": {"type": "mock"},
        }
        with Path("config.yaml").open("w") as f:
            yaml.dump(config, f)

        result = runner.invoke(app, ["check-config", "config.yaml"])
        assert result.exit_code == 1
        assert "Configuration invalid" in result.stdout


def test_run_command(tmp_path: Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Create valid config
        config = {
            "workdir": "tmp",
            "oracle": {"type": "mock"},
            "trainer": {"type": "mock"},
            "dynamics": {"type": "mock"},
            "generator": {"type": "mock"},
            "validator": {"type": "mock"},
            "selector": {"type": "mock"},
        }
        with Path("config.yaml").open("w") as f:
            yaml.dump(config, f)

        result = runner.invoke(app, ["run", "config.yaml"])
        assert result.exit_code == 0
        assert "All components initialized successfully" in result.stdout


def test_run_missing_config(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "nonexistent.yaml"])
    assert result.exit_code == 1
    assert "not found" in result.stdout
