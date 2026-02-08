import pytest
from pathlib import Path
from typing import Any
from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models.config import GlobalConfig
from mlip_autopipec.components.oracle.mock import MockOracle
from mlip_autopipec.core.dataset import Dataset

@pytest.fixture
def mock_config(tmp_path: Path) -> GlobalConfig:
    return GlobalConfig(
        workdir=tmp_path,
        max_cycles=2,
        logging_level="INFO",
        components={
            "generator": {"type": "mock", "n_structures": 5},
            "oracle": {"type": "mock"},
            "trainer": {"type": "mock"},
            "dynamics": {"type": "mock", "selection_rate": 1.0},
            "validator": {"type": "mock"}
        }
    )

def test_full_mock_orchestrator(mock_config: GlobalConfig, tmp_path: Path) -> None:
    # Arrange
    orchestrator = Orchestrator(mock_config)

    # Act
    orchestrator.run()

    # Assert
    # Check cycles
    assert (tmp_path / "cycle_01").exists()
    assert (tmp_path / "cycle_02").exists()
    assert not (tmp_path / "cycle_03").exists()

    # Check potential files
    assert (tmp_path / "cycle_01" / "potential.yace").exists()
    assert (tmp_path / "cycle_02" / "potential.yace").exists()

    # Check dataset
    dataset_path = tmp_path / "dataset.jsonl"
    assert dataset_path.exists()

    # Verify data flow: check if structures have been processed
    # Cycle 1: 5 structures
    # Cycle 2: 5 structures selected from generated
    # Total should be around 10 depending on implementation details
    # We at least check that the file is not empty and contains valid structures
    dataset = Dataset(dataset_path)
    count = 0
    for s in dataset:
        count += 1
        # Verify labeling happened
        assert s.energy is not None
        assert s.forces is not None
        assert s.stress is not None
    assert count > 0

    # Check state
    assert (tmp_path / "workflow_state.json").exists()
    state = orchestrator.state_manager.state
    assert state.current_cycle == 2
    assert state.status == "STOPPED"

class FailingOracle(MockOracle):
    def compute(self, structures: Any) -> Any:
        msg = "Simulated DFT failure"
        raise RuntimeError(msg)

def test_orchestrator_component_failure(mock_config: GlobalConfig) -> None:
    # Inject the failing component via factory override or property patching

    orchestrator = Orchestrator(mock_config)
    orchestrator.oracle = FailingOracle({})

    # Verify graceful failure
    with pytest.raises(RuntimeError, match="Simulated DFT failure"):
        orchestrator.run()

    # State should be updated to ERROR
    # Need to reload state from file to verify persistence
    state_manager = orchestrator.state_manager
    # We can check the in-memory state object as it should be updated
    assert state_manager.state.status == "ERROR"

    # Also verify file persistence
    # Re-instantiate StateManager to read from file
    from mlip_autopipec.core.state import StateManager
    loaded_state = StateManager(mock_config.workdir / "workflow_state.json").state
    assert loaded_state.status == "ERROR"
