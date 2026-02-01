import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from mlip_autopipec.app import app

runner = CliRunner()


def setup_config(path: Path, lammps_cmd: str = "echo"):
    """Helper to create a valid config."""
    config_content = f"""
project_name: "UAT_Project"
potential:
  elements: ["Si"]
  cutoff: 5.0
  pair_style: "hybrid/overlay"
structure_gen:
  strategy: "bulk"
  element: "Si"
  crystal_structure: "diamond"
  lattice_constant: 5.43
md:
  temperature: 300.0
  n_steps: 10
  timestep: 0.001
  ensemble: "NVT"
lammps:
  command: "{lammps_cmd}"
  timeout: 10
  use_mpi: false
"""
    (path / "config.yaml").write_text(config_content)


def test_uat_c02_01_one_shot_success(tmp_path):
    """
    Scenario 2.1: The "One-Shot" MD Run
    """
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        setup_config(Path(td), lammps_cmd="lmp_mock")

        # We need to mock the LammpsRunner execution if we don't have a real LAMMPS
        # Since 'echo' won't produce a dump file, LammpsRunner would fail parsing.
        # We can either:
        # 1. Use a fake script that produces a dump file.
        # 2. Mock the LammpsRunner inside the UAT (less "Black Box" but practical).

        # Let's try to make a fake script if possible, or Mock.
        # Given the constraints, Mocking subprocess in the app context is safer.

        with (
            patch("subprocess.run") as mock_run,
            patch("shutil.which", return_value="/bin/lmp_mock"),
            patch(
                "mlip_autopipec.physics.dynamics.lammps.LammpsRunner._parse_output"
            ) as mock_parse,
        ):
            mock_run.return_value = MagicMock(
                returncode=0, stdout="Simulation Done", stderr=""
            )

            # Mock parsing to return a dummy structure result
            from mlip_autopipec.domain_models.structure import Structure
            import numpy as np

            dummy_struct = Structure(
                symbols=["Si"],
                positions=np.array([[0, 0, 0]]),
                cell=np.eye(3),
                pbc=(True, True, True),
            )
            mock_parse.return_value = (dummy_struct, Path("dump.lammpstrj"), None)

            result = runner.invoke(app, ["run-one-shot", "--config", "config.yaml"])

            assert result.exit_code == 0
            assert "Simulation Completed: Status COMPLETED" in result.stdout
            # Because LammpsRunner creates random dir, just check for existence of base work dir
            # Refactored Orchestrator uses active_learning/iter_001 structure
            assert Path("active_learning/iter_001").exists()


def test_uat_c02_02_missing_executable(tmp_path):
    """
    Scenario 2.2: Missing Executable Handling
    """
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        setup_config(Path(td), lammps_cmd="/path/to/nothing")

        # Here we don't mock, we want it to fail naturally finding the executable
        # BUT shutil.which might check existence.
        # If LammpsRunner checks existence before running, it should fail gracefully.

        result = runner.invoke(app, ["run-one-shot", "--config", "config.yaml"])

        # It should exit with error
        assert result.exit_code != 0
        assert "Executable" in result.stdout and "not found" in result.stdout


if __name__ == "__main__":
    # Allow running this script directly
    sys.exit(pytest.main(["-v", __file__]))
