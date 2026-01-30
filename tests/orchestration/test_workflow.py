import pytest
from unittest.mock import patch

from mlip_autopipec.orchestration.workflow import run_one_shot
from mlip_autopipec.domain_models.config import Config, PotentialConfig, LammpsConfig
from mlip_autopipec.domain_models.job import JobStatus, LammpsResult
from mlip_autopipec.domain_models.structure import Structure


@pytest.fixture
def valid_config():
    return Config(
        project_name="TestProject",
        potential=PotentialConfig(elements=["Si"], cutoff=5.0),
        lammps=LammpsConfig(command="echo", cores=1)
    )


@patch("mlip_autopipec.orchestration.workflow.StructureBuilder")
@patch("mlip_autopipec.orchestration.workflow.LammpsRunner")
def test_run_one_shot_success(MockRunner, MockBuilder, valid_config, sample_ase_atoms):
    # Setup Builder Mock
    mock_builder_instance = MockBuilder.return_value
    mock_structure = Structure.from_ase(sample_ase_atoms)
    mock_builder_instance.build_bulk.return_value = mock_structure
    mock_builder_instance.apply_rattle.return_value = mock_structure

    # Setup Runner Mock
    mock_runner_instance = MockRunner.return_value
    mock_result = LammpsResult(
        job_id="test",
        status=JobStatus.COMPLETED,
        work_dir="/tmp",
        final_structure=mock_structure
    )
    mock_runner_instance.run.return_value = mock_result

    # Execute
    result = run_one_shot(valid_config)

    # Verify interactions
    mock_builder_instance.build_bulk.assert_called_with("Si", "diamond", 5.43)
    mock_builder_instance.apply_rattle.assert_called()
    mock_runner_instance.run.assert_called()

    assert result.status == JobStatus.COMPLETED
