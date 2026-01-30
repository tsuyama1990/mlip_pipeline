from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mlip_autopipec.app import app
from mlip_autopipec.domain_models.structure import Structure

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
        # Create config first. We need to provide required fields for Cycle 02 config.
        # But 'init' creates default template. I need to make sure 'init' template is valid for Cycle 02.
        # Or I manually write config.
        # Let's assume 'init' produces valid config (I should update it too).
        # For now, manually write valid config.
        config_content = """
        project_name: "Test"
        potential:
          elements: ["Si"]
          cutoff: 4.0
        lammps:
          command: "lmp"
        structure_gen:
          element: "Si"
          crystal_structure: "diamond"
          lattice_constant: 5.43
          supercell: [1, 1, 1]
        """
        Path("config.yaml").write_text(config_content)

        result = runner.invoke(app, ["check"])
        assert result.exit_code == 0
        assert "Configuration valid" in result.stdout
        assert Path("mlip_pipeline.log").exists()


def test_check_invalid(tmp_path: Path) -> None:
    """Test 'check' with invalid config."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        p = Path("config.yaml")
        # Invalid cutoff and missing other fields
        p.write_text("project_name: 'Bad'\npotential:\n  cutoff: -1\n  elements: ['A']\nlammps:\n  command: 'echo'\nstructure_gen:\n  element: 'A'\n  crystal_structure: 'sc'\n  lattice_constant: 1.0\n  supercell: [1,1,1]")

        result = runner.invoke(app, ["check"])
        assert result.exit_code == 1
        assert "Validation failed" in result.stdout
        assert "Cutoff must be greater than 0" in result.stdout

def test_run_cycle_02_success(tmp_path: Path) -> None:
    """Test 'run-cycle-02' command success."""
    with runner.isolated_filesystem(temp_dir=tmp_path):
        config_content = """
        project_name: "Test"
        potential:
          elements: ["Si"]
          cutoff: 4.0
        lammps:
          command: "echo"
        structure_gen:
          element: "Si"
          crystal_structure: "diamond"
          lattice_constant: 5.43
          supercell: [1, 1, 1]
        """
        Path("config.yaml").write_text(config_content)

        # Mock dependencies
        # Note: We must patch where they are IMPORTED or DEFINED.
        # StructureGenerator is in modules...
        # io functions are in infrastructure...
        with patch("mlip_autopipec.modules.structure_gen.generator.StructureGenerator.build") as mock_build, \
             patch("mlip_autopipec.infrastructure.io.write_lammps_data") as mock_write, \
             patch("mlip_autopipec.infrastructure.io.run_subprocess") as mock_run, \
             patch("mlip_autopipec.infrastructure.io.read_lammps_dump") as mock_read:

             # Setup mocks
             # Create dummy structure
             import numpy as np
             mock_build.return_value = Structure(
                 symbols=["Si"],
                 positions=np.array([[0,0,0]]),
                 cell=np.eye(3),
                 pbc=(True,True,True)
             )
             mock_run.return_value = ("Simulation log...", "")
             mock_read.return_value = Structure(
                 symbols=["Si"],
                 positions=np.array([[0.1,0,0]]),
                 cell=np.eye(3),
                 pbc=(True,True,True)
             )

             result = runner.invoke(app, ["run-cycle-02"])

             assert result.exit_code == 0
             assert "Simulation Completed" in result.stdout
             assert "COMPLETED" in result.stdout

             # Verify logic
             mock_build.assert_called_once()
             mock_write.assert_called()
             mock_run.assert_called_once()
             mock_read.assert_called_once()
