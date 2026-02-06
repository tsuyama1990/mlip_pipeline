from pathlib import Path
from unittest.mock import MagicMock
import pytest
from mlip_autopipec.config.config_model import GlobalConfig

try:
    from mlip_autopipec.infrastructure.mocks import (
        MockExplorer,
        MockOracle,
        MockTrainer,
        MockValidator,
    )
    from mlip_autopipec.orchestration.orchestrator import Orchestrator
except ImportError:
    Orchestrator = None # type: ignore
    MockExplorer = None # type: ignore
    MockOracle = None # type: ignore
    MockTrainer = None # type: ignore
    MockValidator = None # type: ignore

@pytest.mark.skipif(Orchestrator is None, reason="Orchestrator not implemented yet")
def test_orchestrator_initialization(tmp_path: Path) -> None:
    config = GlobalConfig(work_dir=tmp_path, max_cycles=1)
    mock_explorer = MagicMock()
    mock_oracle = MagicMock()
    mock_trainer = MagicMock()
    mock_validator = MagicMock()
    orch = Orchestrator(config=config, explorer=mock_explorer, oracle=mock_oracle, trainer=mock_trainer, validator=mock_validator)
    assert orch.config.max_cycles == 1

@pytest.mark.skipif(Orchestrator is None, reason="Orchestrator not implemented yet")
def test_orchestrator_run_mock(tmp_path: Path) -> None:
    config = GlobalConfig(work_dir=tmp_path, max_cycles=2)
    orch = Orchestrator(config=config, explorer=MockExplorer(), oracle=MockOracle(), trainer=MockTrainer(), validator=MockValidator())
    orch.run()
    assert len(orch.dataset.structures) > 0
    assert len(orch.dataset.structures) == 2
    assert orch.current_potential is not None
    assert orch.current_cycle == 2
