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
        explorer=ExplorerConfig(type="mock", n_structures=2),
        oracle=OracleConfig(type="mock"),
        trainer=TrainerConfig(type="mock", potential_output_name="mock_potential.yace"),
        validator=ValidatorConfig(type="mock"),
        max_accumulated_structures=100
    )

def test_orchestrator_run(mock_config: GlobalConfig) -> None:
    explorer = MockExplorer(mock_config.explorer)
    oracle = MockOracle()
    trainer = MockTrainer(mock_config.trainer)
    validator = MockValidator(mock_config.validator)

    orchestrator = Orchestrator(mock_config, explorer, oracle, trainer, validator)

    # Run
    orchestrator.run()

    # Assertions
    # Check if accumulated dataset file exists and has content
    dataset_file = mock_config.work_dir / "accumulated_dataset.xyz"
    assert dataset_file.exists()

    # Cycle 1: Explorer 2 -> Oracle 2 -> Write 2
    # Cycle 2: Explorer 2 -> Oracle 2 -> Write 2
    # Total 4 structures should be in the file.
    structures = read(dataset_file, index=":")
    assert len(structures) == 4

    # Check potential path updated to what MockTrainer returns
    assert orchestrator.current_potential_path == Path("mock_potential.yace")

    # Cleanup potential file created in CWD
    if Path("mock_potential.yace").exists():
        Path("mock_potential.yace").unlink()

def test_orchestrator_initial_state(mock_config: GlobalConfig) -> None:
    explorer = MockExplorer(mock_config.explorer)
    oracle = MockOracle()
    trainer = MockTrainer(mock_config.trainer)
    validator = MockValidator(mock_config.validator)

    orchestrator = Orchestrator(mock_config, explorer, oracle, trainer, validator)
    # Check initial potential path
    assert orchestrator.current_potential_path == Path("initial_potential.yace")
    # Check dataset file path setup
    assert orchestrator.dataset_file == mock_config.work_dir / "accumulated_dataset.xyz"

def test_orchestrator_reset(mock_config: GlobalConfig) -> None:
    explorer = MockExplorer(mock_config.explorer)
    oracle = MockOracle()
    trainer = MockTrainer(mock_config.trainer)
    validator = MockValidator(mock_config.validator)

    orchestrator = Orchestrator(mock_config, explorer, oracle, trainer, validator)

    # Create a dummy file to simulate state
    orchestrator.dataset_file.write_text("dummy")
    orchestrator.accumulated_structures_count = 10

    orchestrator.reset()

    assert not orchestrator.dataset_file.exists()
    assert orchestrator.accumulated_structures_count == 0
    assert orchestrator.current_potential_path == Path("initial_potential.yace")

def test_max_accumulated_structures_limit(mock_config: GlobalConfig) -> None:
    # Set limit to 3. n_structures is 2.
    # Cycle 1: 2 structures. Total 2. OK.
    # Cycle 2: 2 structures. Total 4. > 3. Fail.
    mock_config.max_accumulated_structures = 3

    explorer = MockExplorer(mock_config.explorer)
    oracle = MockOracle()
    trainer = MockTrainer(mock_config.trainer)
    validator = MockValidator(mock_config.validator)

    orchestrator = Orchestrator(mock_config, explorer, oracle, trainer, validator)

    with pytest.raises(RuntimeError, match="Max accumulated structures limit exceeded"):
        orchestrator.run()

    # Verify we accumulated 2 from first cycle
    assert orchestrator.accumulated_structures_count == 2
    # Clean up potential
    if Path("mock_potential.yace").exists():
        Path("mock_potential.yace").unlink()
