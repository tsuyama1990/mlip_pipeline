from pathlib import Path
from typing import Any

import pytest

from mlip_autopipec.components.oracle.mock import MockOracle
from mlip_autopipec.core.dataset import Dataset
from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models.config import GlobalConfig, OracleConfig


@pytest.fixture
def mock_config(tmp_path: Path) -> GlobalConfig:
    return GlobalConfig.model_validate({
        "workdir": tmp_path,
        "max_cycles": 2,
        "logging_level": "INFO",
        "components": {
            "generator": {
                "name": "mock",
                "n_structures": 5,
                # Explicitly required params
                "cell_size": 10.0,
                "n_atoms": 2,
                "atomic_numbers": [1, 1]
            },
            "oracle": {"name": "mock"},
            "trainer": {"name": "mock"},
            "dynamics": {
                "name": "mock",
                "selection_rate": 1.0,
                "uncertainty_threshold": 5.0 # Required for mock now
            },
            "validator": {"name": "mock"}
        }
    })

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
    # Use streaming iterator to avoid loading full list
    dataset = Dataset(dataset_path)
    count = 0

    # Streaming iteration: do not use list(dataset)
    for s in dataset:
        count += 1
        # Verify labeling happened
        assert s.energy is not None
        assert s.forces is not None
        assert s.stress is not None
        # Verify integrity
        s.validate_labeled()

    # Cycle 1: 5 structures (selection_rate=1.0)
    # Cycle 2: 5 structures selected from generated
    # Total 10.
    assert count == 10

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
    orchestrator.oracle = FailingOracle(OracleConfig())

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
