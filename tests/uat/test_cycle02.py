import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mlip_autopipec.app import app
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.domain_models.dynamics import LammpsResult
from mlip_autopipec.domain_models.structure import Structure
import numpy as np

runner = CliRunner()


def setup_config(path: Path):
    """Helper to create a valid config."""
    config_content = """
project_name: "UAT_Project"
potential:
  elements: ["Si"]
  cutoff: 5.0
  pair_style: "hybrid/overlay"
  ace_params:
    npot: "FinnisSinclair"
    fs_parameters: [1.0, 1.0, 1.0, 0.5]
    ndensity: 2
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
  command: "lmp"
  timeout: 10
  use_mpi: false
"""
    (path / "config.yaml").write_text(config_content)


def test_uat_c02_01_one_shot_success(tmp_path):
    """
    Scenario 2.1: The "One-Shot" MD Run
    """
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        td_path = Path(td)
        setup_config(td_path)

        # Mock ExplorationPhase to avoid real execution
        with patch("mlip_autopipec.orchestration.workflow.ExplorationPhase") as MockPhase:

            mock_result = LammpsResult(
                job_id="test",
                status=JobStatus.COMPLETED,
                work_dir=td_path,
                duration_seconds=1.0,
                log_content="ok",
                final_structure=Structure(
                    symbols=["Si"], positions=np.array([[0,0,0]]), cell=np.eye(3), pbc=(True,True,True)
                ),
                trajectory_path=td_path / "dump.lammpstrj"
            )
            MockPhase.return_value.execute.return_value = mock_result

            result = runner.invoke(app, ["run-one-shot", "--config", "config.yaml"])

            if result.exit_code != 0:
                print(result.stdout)
                print(result.stderr)

            assert result.exit_code == 0
            assert "Simulation Completed: Status COMPLETED" in result.stdout
            assert "dump.lammpstrj" in result.stdout


def test_uat_c02_02_missing_executable(tmp_path):
    """
    Scenario 2.2: Missing Executable Handling
    """
    # This tests validation logic in LammpsRunner or Config before execution?
    # LammpsConfig doesn't validate existence on init.
    # The runner checks `shutil.which` usually.
    # Since we are mocking ExplorationPhase in the other test, here we might need to test the real runner failure OR mock it failing.
    # The prompt asked for mocking to ensure isolation.

    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        setup_config(Path(td))

        # We can mock ExplorationPhase to raise an error
        with patch("mlip_autopipec.orchestration.workflow.ExplorationPhase") as MockPhase:

            MockPhase.return_value.execute.side_effect = Exception("Executable 'lmp' not found")

            result = runner.invoke(app, ["run-one-shot", "--config", "config.yaml"])

            # It should exit with error
            assert result.exit_code != 0
            assert "Execution failed" in result.stdout
            assert "Executable 'lmp' not found" in result.stdout


if __name__ == "__main__":
    # Allow running this script directly
    sys.exit(pytest.main(["-v", __file__]))
