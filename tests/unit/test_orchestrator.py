from pathlib import Path

import pytest
from ase.io import read

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
        random_seed=42,
        explorer=ExplorerConfig(type="mock", n_structures=2),
        oracle=OracleConfig(type="mock"),
        trainer=TrainerConfig(type="mock", potential_output_name="potential.yace"),
        validator=ValidatorConfig(type="mock"),
    )


def test_orchestrator_initial_state(mock_config: GlobalConfig) -> None:
    explorer = MockExplorer(mock_config.explorer, mock_config.work_dir)
    oracle = MockOracle(mock_config.work_dir)
    trainer = MockTrainer(mock_config.trainer)
    validator = MockValidator(mock_config.validator)

    orchestrator = Orchestrator(mock_config, explorer, oracle, trainer, validator)
    # Check initial potential path
    assert orchestrator.current_potential_path == Path("initial_potential.yace")
    # Check dataset file path setup
    assert orchestrator.dataset_file == mock_config.work_dir / "accumulated_dataset.xyz"


def test_orchestrator_run(mock_config: GlobalConfig) -> None:
    explorer = MockExplorer(mock_config.explorer, mock_config.work_dir)
    oracle = MockOracle(mock_config.work_dir)
    trainer = MockTrainer(mock_config.trainer)
    validator = MockValidator(mock_config.validator)

    orchestrator = Orchestrator(mock_config, explorer, oracle, trainer, validator)
    orchestrator.run()

    # Assertions
    dataset_file = mock_config.work_dir / "accumulated_dataset.xyz"
    assert dataset_file.exists()

    # Cycle 1: Explorer 2 -> Oracle 2 -> Write 2
    # Cycle 2: Explorer 2 -> Oracle 2 -> Write 2
    # Total 4 structures should be in the file.
    structures = read(dataset_file, index=":")
    if not isinstance(structures, list):
        structures = [structures]
    assert len(structures) == 4

    # Check potential path updated.
    # MockTrainer writes to parent of dataset file, which is work_dir.
    # Name is potential.yace
    expected_potential = mock_config.work_dir / "potential.yace"
    assert orchestrator.current_potential_path == expected_potential
    assert expected_potential.exists()
