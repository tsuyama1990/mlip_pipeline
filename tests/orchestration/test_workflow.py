from unittest.mock import patch

import pytest

from mlip_autopipec.domain_models.config import Config, LammpsConfig, PotentialConfig, ExplorationConfig, MDParams, ElementParams
from mlip_autopipec.domain_models.job import JobStatus, LammpsResult
from mlip_autopipec.orchestration.workflow import run_one_shot


@pytest.fixture
def mock_config():
    return Config(
        project_name="Test",
        potential=PotentialConfig(
            elements=["Si"],
            cutoff=2.0,
            element_params={"Si": ElementParams(mass=28.085, lj_sigma=1, lj_epsilon=1, zbl_z=14)}
        ),
        structure_gen=ExplorationConfig(
            composition="Si",
            lattice_constant=5.43,
            md_params=MDParams(temperature=300, n_steps=100)
        ),
        lammps=LammpsConfig()
    )

def test_run_one_shot_success(mock_config, minimal_structure):
    """Test one shot workflow success."""

    with patch("mlip_autopipec.orchestration.workflow.StructureBuilder") as MockBuilder, \
         patch("mlip_autopipec.orchestration.workflow.LammpsRunner") as MockRunner:

        # Setup mocks
        builder_instance = MockBuilder.return_value
        builder_instance.build_bulk.return_value = minimal_structure
        builder_instance.apply_rattle.return_value = minimal_structure

        runner_instance = MockRunner.return_value
        runner_instance.run.return_value = LammpsResult(
            job_id="test",
            status=JobStatus.COMPLETED,
            work_dir=".",
            duration_seconds=1.0,
            log_content="ok",
            final_structure=minimal_structure,
            trajectory_path="traj.dump"
        )

        # Run
        result = run_one_shot(mock_config)

        # Assert
        assert result.status == JobStatus.COMPLETED
        MockBuilder.assert_called()
        MockRunner.assert_called()
        builder_instance.build_bulk.assert_called_with("Si", "diamond", 5.43)
        # Verify passed params
        runner_instance.run.assert_called()
        call_args = runner_instance.run.call_args
        assert call_args[0][1].temperature == 300 # Checked from config
        assert call_args[0][2].elements == ["Si"] # Check potential config passed

def test_run_one_shot_failure(mock_config, minimal_structure):
    """Test one shot workflow failure."""

    with patch("mlip_autopipec.orchestration.workflow.StructureBuilder") as MockBuilder, \
         patch("mlip_autopipec.orchestration.workflow.LammpsRunner") as MockRunner:

        # Setup mocks
        builder_instance = MockBuilder.return_value
        builder_instance.build_bulk.return_value = minimal_structure
        builder_instance.apply_rattle.return_value = minimal_structure # Must return valid structure

        runner_instance = MockRunner.return_value
        runner_instance.run.return_value = LammpsResult(
            job_id="test",
            status=JobStatus.FAILED,
            work_dir=".",
            duration_seconds=1.0,
            log_content="Error",
            final_structure=minimal_structure, # Even failed jobs might return last structure or original
            trajectory_path="traj.dump"
        )

        # Run
        result = run_one_shot(mock_config)

        # Assert
        assert result.status == JobStatus.FAILED
