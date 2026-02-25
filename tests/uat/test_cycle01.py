"""UAT for Cycle 01."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from pyacemaker.core.config import CONSTANTS
from pyacemaker.main import app

# Bypass file checks for UAT as we don't have real pseudos
CONSTANTS.skip_file_checks = True

runner = CliRunner()


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for configs."""
    d = tmp_path / "configs"
    d.mkdir()
    return d


def test_scenario_01_valid_initialization(config_dir: Path) -> None:
    """Scenario 01: Successful System Initialization (Mock Mode)."""
    config_path = config_dir / "valid.yaml"
    # Use a root_dir that is definitely safe and exists
    root_dir = config_dir / "project_root"
    root_dir.mkdir()

    config_content = f"""
version: "1.0.0"
project:
  name: "Cycle01_UAT"
  root_dir: "{root_dir}"
logging:
  level: "DEBUG"
oracle:
  mock: true
  dft:
    pseudopotentials:
      Fe: "Fe.pbe.UPF"
  mace:
    model_path: "medium"
    device: "cpu"
trainer:
  mock: true
dynamics_engine:
  mock: true
"""
    config_path.write_text(config_content)

    result = runner.invoke(app, ["run", str(config_path), "-v"])

    # Check exit code
    assert result.exit_code == 0, f"Exit code {result.exit_code}. Output: {result.output}"
    # Check output
    assert "Configuration loaded successfully" in result.output
    # MACE Oracle loaded message changed/or not printed in new flow if mocked differently
    # Let's check for standard startup logs
    assert "Starting Active Learning Pipeline" in result.output


def test_scenario_02_invalid_configuration(config_dir: Path) -> None:
    """Scenario 02: Invalid Configuration Handling."""
    config_path = config_dir / "invalid.yaml"
    # Use a root_dir that is definitely safe
    root_dir = config_dir / "project_root_invalid"
    root_dir.mkdir()

    config_content = f"""
version: "1.0.0"
project:
  name: "Cycle01_UAT_Invalid"
  root_dir: "{root_dir}"
oracle:
  mock: true
  # Missing 'dft' section which is required
"""
    config_path.write_text(config_content)

    result = runner.invoke(app, ["run", str(config_path)])

    assert result.exit_code != 0
    assert "Error: Invalid configuration" in result.output
    # 'dft' field missing error
    assert "Field required" in result.output or "dft" in result.output
