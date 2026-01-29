from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlip_autopipec.domain_models.config import Config, PotentialConfig
from mlip_autopipec.domain_models.workflow import WorkflowPhase, WorkflowState
from mlip_autopipec.orchestration.protocols import Phase
from mlip_autopipec.orchestration.workflow import WorkflowManager


@pytest.fixture
def basic_config() -> Config:
    return Config(
        project_name="Test",
        potential=PotentialConfig(elements=["Ti"], cutoff=5.0)
    )

def test_workflow_manager_initialization(basic_config: Config) -> None:
    """Test initializing the manager."""
    manager = WorkflowManager(config=basic_config, state_path=Path("state.json"))
    assert manager.config == basic_config
    assert manager.state.cycle_index == 0
    assert manager.state.current_phase == WorkflowPhase.EXPLORATION

def test_workflow_manager_run_cycle(basic_config: Config, tmp_path: Path) -> None:
    """Test running a cycle with a mock phase."""
    state_path = tmp_path / "state.json"
    manager = WorkflowManager(config=basic_config, state_path=state_path)

    # Mock Phase
    mock_phase = MagicMock(spec=Phase)
    mock_phase.name = "Exploration"
    # Phase returns updated state (moved to SELECTION manually in this mock logic?)
    # Wait, manager handles transition if phases are standard.
    # But manager calls execution. The mock should return the state.

    def execute_side_effect(state: WorkflowState, config: Config) -> WorkflowState:
        # Simulate work
        state.is_halted = False
        return state

    mock_phase.execute.side_effect = execute_side_effect

    manager.register_phase(mock_phase, WorkflowPhase.EXPLORATION)

    # Run
    manager.run_cycle()

    # Verify phase execution
    mock_phase.execute.assert_called_once()

    # Verify transition
    # Exploration -> Selection
    assert manager.state.current_phase == WorkflowPhase.SELECTION

    # Verify state saved
    assert state_path.exists()
    assert "SELECTION" in state_path.read_text()
