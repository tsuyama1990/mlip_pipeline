from pathlib import Path
from unittest.mock import patch

import yaml
from typer.testing import CliRunner

from mlip_autopipec.app import app

runner = CliRunner()

def test_init_command(tmp_path: Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert Path("config.yaml").exists()

        # Verify content
        with Path("config.yaml").open() as f:
            config = yaml.safe_load(f)
            # Depending on the template, we check a key
            assert "project_name" in config

def test_init_command_existing(tmp_path: Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("config.yaml").touch()
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 1
        assert "already exists" in result.stdout

def test_check_command(tmp_path: Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(app, ["init"])
        result = runner.invoke(app, ["check"])
        assert result.exit_code == 0
        assert "valid" in result.stdout.lower()

def test_check_command_missing(tmp_path: Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["check"])
        assert result.exit_code == 1
        assert "not found" in result.stdout

def test_run_loop_mocked(tmp_path: Path) -> None:
     with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(app, ["init"])
        # Mock WorkflowManager to prevent actual execution
        with patch("mlip_autopipec.cli.commands.WorkflowManager") as mock_wm:
            result = runner.invoke(app, ["run-loop"])
            assert result.exit_code == 0
            mock_wm.assert_called_once()
            mock_wm.return_value.run.assert_called_once()

def test_run_loop_missing(tmp_path: Path) -> None:
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["run-loop"])
        assert result.exit_code == 1
        assert "not found" in result.stdout
