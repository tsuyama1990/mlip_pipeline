from unittest.mock import patch, MagicMock

import pytest
from mlip_autopipec.domain_models.config import Config, LammpsConfig, PotentialConfig
from mlip_autopipec.orchestration.workflow import run_one_shot


@pytest.fixture
def config():
    return Config(
        project_name="TestProject",
        potential=PotentialConfig(elements=["Si"], cutoff=5.0),
        lammps=LammpsConfig()
    )


@patch("mlip_autopipec.orchestration.workflow.StructureBuilder")
@patch("mlip_autopipec.orchestration.workflow.LammpsRunner")
def test_run_one_shot_flow(mock_runner_cls, mock_builder_cls, config):
    # Mock builder
    mock_builder = mock_builder_cls.return_value
    mock_builder.build_bulk.return_value = "structure"
    mock_builder.apply_rattle.return_value = "rattled_structure"

    # Mock runner
    mock_runner = mock_runner_cls.return_value

    # Mock result with proper Structure object for final_structure
    mock_final_structure = MagicMock()
    mock_final_structure.positions = [1, 2, 3] # Simulate positions list/array

    mock_result = MagicMock()
    mock_result.status = "COMPLETED" # Enum equality works with string
    mock_result.final_structure = mock_final_structure
    mock_result.log_content = ""

    mock_runner.run.return_value = mock_result

    run_one_shot(config)

    # Assertions to check flow
    mock_builder.build_bulk.assert_called()
    mock_builder.apply_rattle.assert_called()
    mock_runner.run.assert_called()
