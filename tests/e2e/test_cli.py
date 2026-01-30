from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mlip_autopipec.app import app
from mlip_autopipec.domain_models.job import JobStatus, LammpsResult

runner = CliRunner()

def test_init(tmp_path: Path) -> None:
    """Test 'init' command."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert "Created template configuration" in result.stdout
        assert Path("config.yaml").exists()

def test_init_existing(tmp_path: Path) -> None:
    """Test 'init' fails if file exists."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("config.yaml").touch()
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 1
        assert "already exists" in result.stdout

def test_init_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 'init' handles exceptions during write."""
    from mlip_autopipec.infrastructure import io

    def mock_dump(*args: Any, **kwargs: Any) -> None:
        msg = "Permission denied"
        raise OSError(msg)

    monkeypatch.setattr(io, "dump_yaml", mock_dump)

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 1
        assert "Failed to create config: Permission denied" in result.stdout

def test_check_valid(tmp_path: Path) -> None:
    """Test 'check' with valid config."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Create config first
        runner.invoke(app, ["init"])

        result = runner.invoke(app, ["check"])
        assert result.exit_code == 0
        assert "Configuration valid" in result.stdout
        assert Path("mlip_pipeline.log").exists()

def test_check_invalid(tmp_path: Path) -> None:
    """Test 'check' with invalid config."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        p = Path("config.yaml")
        p.write_text("project_name: 'Bad'\npotential:\n  cutoff: -1\n  elements: ['A']")

        result = runner.invoke(app, ["check"])
        assert result.exit_code == 1
        assert "Validation failed" in result.stdout
        assert "Cutoff must be greater than 0" in result.stdout

def test_run_cycle_02_success(tmp_path: Path):
    """Test run-cycle-02 command success."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(app, ["init"])

        with patch("mlip_autopipec.cli.commands.run_one_shot") as mock_run:
            mock_run.return_value = LammpsResult(
                job_id="test",
                status=JobStatus.COMPLETED,
                work_dir=Path("."),
                duration_seconds=1.0
            )
            result = runner.invoke(app, ["run-cycle-02"])
            assert result.exit_code == 0
            assert "Simulation Completed: Status COMPLETED" in result.stdout

def test_run_cycle_02_missing_config(tmp_path: Path):
    """Test run-cycle-02 command fails without config."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["run-cycle-02"])
        assert result.exit_code == 1
        assert "not found" in result.stdout
