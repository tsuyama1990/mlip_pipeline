from pathlib import Path
from unittest.mock import patch

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
    with (
        patch("mlip_autopipec.orchestration.orchestrator.StateManager") as MockState,
        patch("mlip_autopipec.orchestration.orchestrator.PacemakerTrainer") as MockTrainer,
    ):
        orch = Orchestrator(mock_config)
        assert orch.config == mock_config
        MockState.assert_called_once()
        MockTrainer.assert_called_once()


def test_orchestrator_run_loop(mock_config: Config) -> None:
    with (
        patch("mlip_autopipec.orchestration.orchestrator.StateManager") as MockState,
        patch("mlip_autopipec.orchestration.orchestrator.PacemakerTrainer") as MockTrainer,
    ):
        # Setup mocks
        mock_state_instance = MockState.return_value
        # We need a state that updates.
        # Since logic is in orchestrator, we can assume it updates the state object.
        # But load() is called once at start.
        mock_state_instance.load.return_value = WorkflowState(iteration=0)

        mock_trainer_instance = MockTrainer.return_value
        mock_trainer_instance.train.return_value = Path("output.yace")

        orch = Orchestrator(mock_config)
        orch.run()

        # 0 < 2 -> Run -> iteration becomes 1
        # 1 < 2 -> Run -> iteration becomes 2
        # 2 < 2 -> Stop.
        assert mock_trainer_instance.train.call_count == 2
        assert mock_state_instance.save.call_count >= 2
