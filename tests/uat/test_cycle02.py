import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mlip_autopipec.app import app
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.domain_models.dynamics import LammpsResult
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.exploration import ExplorationTask
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
                trajectory_path=td_path / "dump.extxyz"
            )
            MockPhase.return_value.execute.return_value = mock_result

            result = runner.invoke(app, ["run-one-shot", "--config", "config.yaml"])

            if result.exit_code != 0:
                print(result.stdout)
                print(result.stderr)

            assert result.exit_code == 0
            assert "Simulation Completed: Status COMPLETED" in result.stdout
            assert "dump.extxyz" in result.stdout


def test_uat_c02_02_missing_executable(tmp_path):
    """
    Scenario 2.2: Missing Executable Handling
    """
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        setup_config(Path(td))

        # We need to force MD execution to test LammpsRunner executable check.
        # By default run-one-shot uses Cycle 0 (Static).
        # We patch AdaptivePolicy to return MD task.

        with patch("mlip_autopipec.orchestration.phases.exploration.AdaptivePolicy") as MockPolicy:
             MockPolicy.return_value.decide.return_value = ExplorationTask(
                 method="MD", modifiers=[]
             )

             # Mock shutil.which to return None (simulating missing executable)
             with patch("mlip_autopipec.physics.dynamics.lammps.shutil.which", return_value=None):

                 # Structure generation might still happen, assume it works or mock it if needed.
                 # Actually LammpsRunner init checks command validity (regex) but shutil.which is checked at execution time?
                 # No, LammpsRunner now checks `shutil.which` at _execute (MD runtime).

                 result = runner.invoke(app, ["run-one-shot", "--config", "config.yaml"])

                 # It should exit with error
                 assert result.exit_code != 0
                 assert "Execution failed" in result.stdout
                 assert "Executable 'lmp' not found" in result.stdout


if __name__ == "__main__":
    # Allow running this script directly
    sys.exit(pytest.main(["-v", __file__]))
