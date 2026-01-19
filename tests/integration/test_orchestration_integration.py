import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.config.models import SystemConfig
from mlip_autopipec.orchestration.manager import WorkflowManager
from mlip_autopipec.orchestration.models import OrchestratorConfig


@pytest.fixture
def mock_config(tmp_path: Path) -> MagicMock:
    config = MagicMock(spec=SystemConfig)
    config.working_dir = tmp_path
    config.db_path = tmp_path / "test.db"
    # Explicitly set optional configs to avoid AttributeError if spec fails or logical None checks
    config.surrogate_config = MagicMock()
    config.dft_config = MagicMock()
    config.training_config = MagicMock()
    config.inference_config = MagicMock()
    return config


@pytest.fixture
def mock_orch_config() -> OrchestratorConfig:
    return OrchestratorConfig(max_generations=2, workers=1, dask_scheduler_address=None)


def test_workflow_grand_mock(mock_config: MagicMock, mock_orch_config: OrchestratorConfig) -> None:
    """
    Simulate a full run where we mock the underlying components (Generator, DFT, etc.)
    but allow the WorkflowManager logic to execute its control flow.
    """
    with (
        patch("mlip_autopipec.orchestration.manager.DatabaseManager") as MockDB,
        patch("mlip_autopipec.orchestration.manager.TaskQueue"),
        patch("mlip_autopipec.orchestration.manager.Dashboard"),
        patch("mlip_autopipec.orchestration.manager.StructureBuilder") as MockBuilder,
        patch("mlip_autopipec.orchestration.manager.SurrogatePipeline") as MockSurrogate,
        patch("mlip_autopipec.orchestration.manager.QERunner") as MockQE,
        patch("mlip_autopipec.orchestration.manager.DatasetBuilder"),
        patch("mlip_autopipec.orchestration.manager.TrainConfigGenerator"),
        patch("mlip_autopipec.orchestration.manager.PacemakerWrapper") as MockPacemaker,
        patch("mlip_autopipec.orchestration.manager.LammpsRunner"),
    ):
        # Configure mock DB
        MockDB.return_value.count.return_value = 100

        # Setup Component Mocks to simulate data flow
        # Exploration
        mock_candidates = [MagicMock(), MagicMock()]
        MockBuilder.return_value.build.return_value = mock_candidates
        # Surrogate returns subset
        MockSurrogate.return_value.run.return_value = (mock_candidates, MagicMock())

        # DFT
        # We don't need to mock return values complexly because TaskQueue is also mocked
        # (or rather, the manager calls task_queue.submit/wait).
        # We need to ensure we don't crash.

        manager = WorkflowManager(mock_config, mock_orch_config)

        # Run the manager
        manager.run()

        # Assertions

        # 1. Check Phases Executed
        # Generation 0 and 1 -> 2 loops.
        assert manager.state.current_generation == 2

        # 2. Check Exploration calls
        assert MockBuilder.return_value.build.call_count == 2
        assert MockSurrogate.return_value.run.call_count == 2

        # 3. Check DFT calls
        assert MockQE.call_count == 2  # Initialized once per phase

        # 4. Check Training calls
        assert MockPacemaker.call_count == 2

        # 5. Check Dashboard updated
        assert manager.dashboard.update.call_count >= 2  # type: ignore

        # 6. Verify state file persisted
        state_file = mock_config.working_dir / "workflow_state.json"
        assert state_file.exists()
        final_state = json.loads(state_file.read_text())
        assert final_state["current_generation"] == 2
