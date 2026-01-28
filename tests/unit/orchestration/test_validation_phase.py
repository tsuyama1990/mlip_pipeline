from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.validation import ValidationConfig
from mlip_autopipec.domain_models.state import WorkflowState
from mlip_autopipec.domain_models.validation import ValidationResult
from mlip_autopipec.orchestration.phases.validation import ValidationPhase


@pytest.fixture
def mock_manager(tmp_path):
    manager = MagicMock()
    manager.work_dir = tmp_path / "work"
    manager.work_dir.mkdir()

    manager.state = WorkflowState()
    pot_path = manager.work_dir / "pot.yace"
    pot_path.touch()
    manager.state.latest_potential_path = pot_path

    # Mocking deep config structure
    manager.config.target_system.elements = ["Fe"]
    manager.config.target_system.crystal_structure = "bcc"
    manager.config.validation_config = ValidationConfig()
    return manager

@patch("mlip_autopipec.orchestration.phases.validation.ValidationRunner")
@patch("mlip_autopipec.orchestration.phases.validation.bulk")
def test_validation_phase_execution(mock_bulk, mock_runner_class, mock_manager):
    # Setup
    mock_runner_instance = mock_runner_class.return_value
    mock_runner_instance.run.return_value = [
        ValidationResult(module="phonon", passed=True),
        ValidationResult(module="elastic", passed=True)
    ]

    mock_atoms = Atoms("Fe")
    mock_bulk.return_value = mock_atoms

    # Execute
    phase = ValidationPhase(mock_manager)
    phase.execute()

    # Verify
    mock_runner_class.assert_called_once_with(
        config=mock_manager.config.validation_config,
        work_dir=mock_manager.work_dir / "validation_gen_0"
    )
    mock_runner_instance.run.assert_called_once_with(mock_atoms, mock_manager.state.latest_potential_path)

@patch("mlip_autopipec.orchestration.phases.validation.ValidationRunner")
def test_validation_skips_if_no_potential(mock_runner_class, mock_manager):
    mock_manager.state.latest_potential_path = None

    phase = ValidationPhase(mock_manager)
    phase.execute()

    mock_runner_class.assert_not_called()
