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
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        setup_config(Path(td))

        # IMPORTANT: We want to test that if LammpsRunner checks for the executable and fails,
        # the error is propagated.
        # However, `run-one-shot` uses `ExplorationPhase`.
        # `ExplorationPhase` instantiates `LammpsRunner`.
        # `LammpsRunner` checks `validate_command` on init, and `shutil.which` during `_execute` (or maybe init?).
        # My `LammpsRunner` code shows `validate_command` in init, but `shutil.which` in `_execute`.

        # So we allow `ExplorationPhase` to run, but we mock `LammpsRunner._execute` or `shutil.which`.
        # If we mock `ExplorationPhase`, we are not testing the runner's behavior.

        # Strategy: Allow ExplorationPhase to run logic, but mock internal subprocess/shutil.which
        # This requires not mocking ExplorationPhase class, but its dependencies.

        with patch("mlip_autopipec.physics.dynamics.lammps.shutil.which", return_value=None):
             # Also assume StructureGenFactory works (it generates structure)
             # But running actual ExplorationPhase requires StructureGen.
             # Let's assume default config works for generation.

             # We might need to mock StructureGenFactory if it's too slow/complex, but for bulk it's fast.

             result = runner.invoke(app, ["run-one-shot", "--config", "config.yaml"])

             # It should exit with error
             assert result.exit_code != 0
             assert "Execution failed" in result.stdout
             # The error message from LammpsRunner._execute should be "Executable ... not found"
             # OR FileNotFoundError caught and re-raised.

             # Wait, ExplorationPhase calls runner.run(). runner.run() calls _execute().
             # _execute raises FileNotFoundError if shutil.which is None.
             # runner.run catches Exception and returns LammpsResult(status=FAILED, log_content=str(e)).
             # So run_one_shot gets FAILED result.
             # command.py raises Exit(1) if status != COMPLETED.

             # Check output for "Executable 'lmp' not found" (log content is printed on failure)
             assert "Executable 'lmp' not found" in result.stdout


if __name__ == "__main__":
    # Allow running this script directly
    sys.exit(pytest.main(["-v", __file__]))
