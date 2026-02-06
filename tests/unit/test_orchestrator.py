from pathlib import Path

import pytest

from mlip_autopipec.config import (
    ExplorerConfig,
    GlobalConfig,
    OracleConfig,
    TrainerConfig,
    ValidatorConfig,
)
from mlip_autopipec.infrastructure.mocks import MockExplorer, MockOracle, MockTrainer, MockValidator
from mlip_autopipec.orchestration.orchestrator import Orchestrator


@pytest.fixture
def mock_config(tmp_path: Path) -> GlobalConfig:
    return GlobalConfig(
        work_dir=tmp_path,
        max_cycles=2,
        explorer=ExplorerConfig(type="mock"),
        oracle=OracleConfig(type="mock"),
        trainer=TrainerConfig(type="mock", potential_output_name="mock_potential.yace"),
        validator=ValidatorConfig(type="mock")
    )

def test_orchestrator_run(mock_config: GlobalConfig) -> None:
    explorer = MockExplorer()
    oracle = MockOracle()
    trainer = MockTrainer(mock_config.trainer)
    validator = MockValidator(mock_config.validator)

    orchestrator = Orchestrator(mock_config, explorer, oracle, trainer, validator)

    # Run
    orchestrator.run()

    # Assertions
    # Check if accumulated dataset file exists and has content
    dataset_file = mock_config.work_dir / mock_config.dataset_file_name
    assert dataset_file.exists()

    # Cycle 1: Explorer 2 -> Oracle 2 -> Write 2
    # Cycle 2: Explorer 2 -> Oracle 2 -> Write 2
    # Total 4 structures should be in the file.
    # Since we use ASE write with append, checking file size > 0 is basic check.
    # To check exact count we would need to read it back, but ASE dependency might be heavy for unit test?
    # No, we have ASE installed.
    from ase.io import read
    structures = read(dataset_file, index=":")
    assert len(structures) == 4

    # Check potential path updated to what MockTrainer returns
    # MockTrainer writes to "mock_potential.yace" in CWD (bad practice in Mock? MockTrainer uses config name).
    # In MockTrainer implementation I used: potential_path = Path(potential_filename) which is relative to CWD.
    # Orchestrator uses it.
    assert orchestrator.current_potential_path == Path("mock_potential.yace")

    # Cleanup potential file created in CWD
    if Path("mock_potential.yace").exists():
        Path("mock_potential.yace").unlink()

def test_orchestrator_initial_state(mock_config: GlobalConfig) -> None:
    explorer = MockExplorer()
    oracle = MockOracle()
    trainer = MockTrainer(mock_config.trainer)
    validator = MockValidator(mock_config.validator)

    orchestrator = Orchestrator(mock_config, explorer, oracle, trainer, validator)
    # Check initial potential path
    assert orchestrator.current_potential_path == Path("initial_potential.yace")
    # Check dataset file path setup
    assert orchestrator.dataset_file == mock_config.work_dir / mock_config.dataset_file_name
