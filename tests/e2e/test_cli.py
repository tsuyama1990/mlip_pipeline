from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from mlip_autopipec.app import app
from mlip_autopipec.domain_models.job import JobResult, JobStatus

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

@patch("mlip_autopipec.cli.commands.workflow.run_one_shot")
def test_run_cycle_02_success(mock_run, tmp_path):
    """Test run-cycle-02 command success."""
    # Setup mock return
    mock_result = JobResult(
        job_id="test_job",
        status=JobStatus.COMPLETED,
        work_dir=Path("/tmp/work"),
        duration_seconds=1.0,
        log_content="Success"
    )
    mock_run.return_value = mock_result

    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(app, ["init"])
        result = runner.invoke(app, ["run-cycle-02"])

        assert result.exit_code == 0
        assert "Simulation Completed: Status COMPLETED" in result.stdout

@patch("mlip_autopipec.cli.commands.workflow.run_one_shot")
def test_run_cycle_02_failure(mock_run, tmp_path):
    """Test run-cycle-02 command failure."""
    # Setup mock return
    mock_result = JobResult(
        job_id="test_job",
        status=JobStatus.FAILED,
        work_dir=Path("/tmp/work"),
        duration_seconds=1.0,
        log_content="Kaboom"
    )
    mock_run.return_value = mock_result

    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(app, ["init"])
        result = runner.invoke(app, ["run-cycle-02"])

        assert result.exit_code == 1
        assert "Simulation Failed: Status FAILED" in result.stdout
        assert "Kaboom" in result.stdout
