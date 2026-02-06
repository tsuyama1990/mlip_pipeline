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
    # Pass work_dir to mocks as expected now
    oracle = MockOracle(work_dir=mock_config.work_dir)
    trainer = MockTrainer(mock_config.trainer, work_dir=mock_config.work_dir)
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
    from ase.io import read
    structures = read(dataset_file, index=":")
    assert len(structures) == 4

    # Check potential path updated to what MockTrainer returns
    # MockTrainer now writes to work_dir / potential_output_name
    expected_potential_path = mock_config.work_dir / "mock_potential.yace"

    assert orchestrator.current_potential_path == expected_potential_path
    assert expected_potential_path.exists()

def test_orchestrator_initial_state(mock_config: GlobalConfig) -> None:
    explorer = MockExplorer()
    oracle = MockOracle(work_dir=mock_config.work_dir)
    trainer = MockTrainer(mock_config.trainer, work_dir=mock_config.work_dir)
    validator = MockValidator(mock_config.validator)

    orchestrator = Orchestrator(mock_config, explorer, oracle, trainer, validator)
    # Check initial potential path defaults to work_dir/initial_potential.yace
    assert orchestrator.current_potential_path == mock_config.work_dir / "initial_potential.yace"
    # Check dataset file path setup
    assert orchestrator.dataset_file == mock_config.work_dir / mock_config.dataset_file_name
