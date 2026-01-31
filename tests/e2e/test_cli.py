from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mlip_autopipec.app import app

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


def test_validate_command(tmp_path: Path) -> None:
    """Test 'validate' command."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(app, ["init"])
        pot = Path("pot.yace")
        pot.touch()

        with patch("mlip_autopipec.cli.commands.Orchestrator") as MockOrch:
            result = runner.invoke(app, ["validate", "--potential", "pot.yace"])

            assert result.exit_code == 0
            assert "Validation Completed" in result.stdout
            MockOrch.return_value.validate_potential.assert_called()


def test_validate_missing_potential(tmp_path: Path) -> None:
    """Test 'validate' fails without potential."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(app, ["init"])

        result = runner.invoke(app, ["validate"])
        # Should fail if not in config
        assert result.exit_code == 1
        assert "Potential file" in result.stdout
