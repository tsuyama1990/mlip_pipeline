from pathlib import Path
from unittest.mock import MagicMock, patch

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
    explorer = MagicMock()
    oracle = MagicMock()
    trainer = MagicMock()

    with (
        patch("mlip_autopipec.orchestration.orchestrator.StateManager") as MockState,
    ):
        orch = Orchestrator(mock_config, explorer, oracle, trainer)
        assert orch.config == mock_config
        MockState.assert_called_once()


def test_orchestrator_run_loop(mock_config: Config) -> None:
    explorer = MagicMock()
    oracle = MagicMock()
    trainer = MagicMock()

    with (
        patch("mlip_autopipec.orchestration.orchestrator.StateManager") as MockState,
    ):
        # Setup mocks
        mock_state_instance = MockState.return_value
        mock_state_instance.load.return_value = WorkflowState(iteration=0)

        trainer.train.return_value = Path("output.yace")

        # Explorer and Oracle mocks should return something
        explorer.explore.return_value = {"status": "ok"}
        oracle.compute.return_value = {"status": "ok"}

        orch = Orchestrator(mock_config, explorer, oracle, trainer)
        orch.run()

        # 0 < 2 -> Run -> iteration becomes 1
        # 1 < 2 -> Run -> iteration becomes 2
        # 2 < 2 -> Stop.
        assert trainer.train.call_count == 2
        assert mock_state_instance.save.call_count >= 2
        assert explorer.explore.call_count == 2
        assert oracle.compute.call_count == 2
