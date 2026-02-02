from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from mlip_autopipec.config import Config, OrchestratorConfig, ProjectConfig, TrainingConfig
from mlip_autopipec.domain_models.workflow import WorkflowState
from mlip_autopipec.orchestration.orchestrator import Orchestrator


@pytest.fixture
def mock_config(temp_dir: Path) -> Config:
    (temp_dir / "data.pckl").touch()
    return Config(
        project=ProjectConfig(name="Test"),
        training=TrainingConfig(dataset_path=temp_dir / "data.pckl"),
        orchestrator=OrchestratorConfig(max_iterations=2),
    )


def test_orchestrator_initialization(mock_config: Config) -> None:
    # Create mocks for dependencies
    mock_explorer = Mock()
    mock_oracle = Mock()
    mock_trainer = Mock()
    mock_validator = Mock()

    with patch("mlip_autopipec.orchestration.orchestrator.StateManager") as MockState:
        orch = Orchestrator(
            config=mock_config,
            explorer=mock_explorer,
            oracle=mock_oracle,
            trainer=mock_trainer,
            validator=mock_validator
        )
        assert orch.config == mock_config
        MockState.assert_called_once()


def test_orchestrator_run_loop(mock_config: Config) -> None:
    mock_explorer = Mock()
    mock_oracle = Mock()
    mock_trainer = Mock()
    mock_validator = Mock()

    with patch("mlip_autopipec.orchestration.orchestrator.StateManager") as MockState:
        # Setup state
        mock_state_instance = MockState.return_value
        mock_state_instance.load.return_value = WorkflowState(iteration=0)

        # Setup component returns
        # Must return candidates to avoid convergence check breaking the loop
        mock_explorer.explore.return_value = {"halted": True, "candidates": [Path("cand.xyz")]}
        mock_oracle.compute.return_value = [Path("new_data.pckl")]
        mock_trainer.train.return_value = Path("output.yace")
        mock_validator.validate.return_value = {"passed": True}

        orch = Orchestrator(
            config=mock_config,
            explorer=mock_explorer,
            oracle=mock_oracle,
            trainer=mock_trainer,
            validator=mock_validator
        )
        orch.run()

        # 0 < 2 -> Run -> iteration becomes 1
        # 1 < 2 -> Run -> iteration becomes 2
        # 2 < 2 -> Stop.
        assert mock_trainer.train.call_count == 2
        assert mock_state_instance.save.call_count >= 2

        # Verify interactions
        assert mock_explorer.explore.call_count == 2
        assert mock_oracle.compute.call_count == 2
        assert mock_validator.validate.call_count == 2
