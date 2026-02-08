from pathlib import Path
from typing import Any

import pytest

from mlip_autopipec.components.oracle.mock import MockOracle
from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models.config import GlobalConfig


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
            "validator": {"type": "mock"},
        },
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
    assert (tmp_path / "dataset.jsonl").exists()

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
    # Since Orchestrator init calls factory, we can modify the factory or
    # instantiate orchestrator and then replace the component.

    orchestrator = Orchestrator(mock_config)
    orchestrator.oracle = FailingOracle({})

    # Verify graceful failure
    with pytest.raises(RuntimeError, match="Simulated DFT failure"):
        orchestrator.run()

    # State should be updated to ERROR
    assert orchestrator.state_manager.state.status == "ERROR"
