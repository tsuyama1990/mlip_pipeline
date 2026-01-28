from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml
from ase import Atoms
from typer.testing import CliRunner

from mlip_autopipec.app import app
from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.dft.runner import QERunner
from mlip_autopipec.domain_models.dft_models import DFTResult

runner = CliRunner()


def test_init_creates_template(tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert (Path("input.yaml")).exists()

        # Verify content
        with Path("input.yaml").open() as f:
            data = yaml.safe_load(f)
        assert "target_system" in data
        assert "dft" in data


def test_init_existing_file(tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Create dummy file
        Path("input.yaml").touch()
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert "input.yaml already exists" in result.stdout


def test_validate_valid_config(tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Create valid config
        runner.invoke(app, ["init"])

        # Override pseudopotential_dir to a valid path for validation
        with Path("input.yaml").open() as f:
            data = yaml.safe_load(f)

        pseudo_dir = tmp_path / "upf"
        pseudo_dir.mkdir()
        (pseudo_dir / "Fe.upf").touch()
        data["dft"]["pseudopotential_dir"] = str(pseudo_dir)

        with Path("input.yaml").open("w") as f:
            yaml.dump(data, f)

        result = runner.invoke(app, ["validate", "input.yaml"])
        assert result.exit_code == 0
        assert "Validation Successful" in result.stdout


def test_validate_invalid_config(tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with Path("bad_config.yaml").open("w") as f:
            f.write("target_system: []\n")  # Invalid type

        result = runner.invoke(app, ["validate", "bad_config.yaml"])
        assert result.exit_code == 1
        assert "Validation Error" in result.stdout


def test_validate_missing_file(tmp_path):
    result = runner.invoke(app, ["validate", "non_existent.yaml"])
    assert result.exit_code == 1
    # Error message might vary depending on OS error or my code, but usually "Error"
    assert "Error" in result.stdout or "No such file" in result.stdout


def test_db_init(tmp_path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(app, ["init"])

        # Fix path
        with Path("input.yaml").open() as f:
            data = yaml.safe_load(f)

        pseudo_dir = tmp_path / "upf"
        pseudo_dir.mkdir()
        (pseudo_dir / "Fe.upf").touch()
        data["dft"]["pseudopotential_dir"] = str(pseudo_dir)

        with Path("input.yaml").open("w") as f:
            yaml.dump(data, f)

        result = runner.invoke(app, ["db", "init", "--config", "input.yaml"])
        assert result.exit_code == 0
        assert "Database initialized" in result.stdout
        assert Path("mlip.db").exists()


@patch("shutil.which")
@patch("subprocess.run")
@patch("mlip_autopipec.dft.runner.QEOutputParser")
def test_oracle_flow(mock_parser_class, mock_run, mock_which, tmp_path):
    """
    Scenario: Run a static calculation on Silicon (Mocked)
    """
    # GIVEN a configuration and structure
    p_dir = tmp_path / "pseudos"
    p_dir.mkdir()
    (p_dir / "Si.upf").touch()

    config = DFTConfig(
        command="pw.x", pseudopotential_dir=p_dir, pseudopotentials={"Si": "Si.upf"}, kspacing=0.1
    )

    mock_which.return_value = "/bin/pw.x"

    runner = QERunner(config=config, work_dir=tmp_path, parser_class=mock_parser_class)
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)

    # Mock subprocess success
    mock_process = MagicMock(returncode=0)
    mock_process.stderr = ""  # Ensure stderr is a string
    mock_run.return_value = mock_process

    # Mock Parser
    mock_parser_instance = mock_parser_class.return_value
    mock_parser_instance.parse.return_value = DFTResult(
        uid="test",
        energy=-135.0,
        forces=[[0.0, 0.0, 0.0]],
        stress=np.zeros((3, 3)).tolist(),
        succeeded=True,
        converged=True,
        wall_time=1.0,
        parameters={},
    )

    # WHEN I run compute
    result = runner.run(atoms)

    # THEN
    if not result.converged:
        pytest.fail(f"DFT failed with error: {result.error_message}")

    assert result.energy == -135.0
    assert result.converged is True
    # Verify input was written
    mock_run.assert_called_once()

    # Verify command was "securely" executed (shell=False)
    args, kwargs = mock_run.call_args
    assert kwargs.get("shell") is False
    assert "pw.x" in args[0] or "pw.x" in args[0][0]
