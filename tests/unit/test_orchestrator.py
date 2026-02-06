import pytest
from pathlib import Path
from mlip_autopipec.orchestration.orchestrator import Orchestrator
from mlip_autopipec.config import GlobalConfig, ExplorerConfig, OracleConfig, TrainerConfig
from mlip_autopipec.infrastructure.mocks import MockExplorer, MockOracle, MockTrainer, MockValidator
from mlip_autopipec.domain_models import Dataset

@pytest.fixture
def mock_config(tmp_path: Path) -> GlobalConfig:
    return GlobalConfig(
        work_dir=tmp_path,
        max_cycles=2,
        explorer=ExplorerConfig(type="mock"),
        oracle=OracleConfig(type="mock"),
        trainer=TrainerConfig(type="mock")
    )

def test_orchestrator_run(mock_config: GlobalConfig) -> None:
    explorer = MockExplorer()
    oracle = MockOracle()
    trainer = MockTrainer()
    validator = MockValidator()

    orchestrator = Orchestrator(mock_config, explorer, oracle, trainer, validator)

    # Run
    orchestrator.run()

    # Assertions
    # Cycle 1: Explorer produces 2, Oracle labels 2. Dataset len = 2.
    # Cycle 2: Explorer produces 2, Oracle labels 2. Dataset len = 4.
    assert len(orchestrator.dataset.structures) == 4

    # Check potential path updated
    assert orchestrator.current_potential_path == Path("mock_potential.yace")

def test_orchestrator_initial_state(mock_config: GlobalConfig) -> None:
    explorer = MockExplorer()
    oracle = MockOracle()
    trainer = MockTrainer()
    validator = MockValidator()

    orchestrator = Orchestrator(mock_config, explorer, oracle, trainer, validator)
    assert len(orchestrator.dataset.structures) == 0
