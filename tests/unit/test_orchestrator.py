from pathlib import Path
from unittest.mock import MagicMock, patch

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
    """
    Tests that the Orchestrator initializes correctly with the given configuration.
    """
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
    """
    Tests the full execution cycle of the Orchestrator using mock components.
    Verifies that structures are accumulated and potential files are created.
    """
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


@patch("mlip_autopipec.orchestration.orchestrator.iread")
def test_orchestrator_uses_streaming(mock_iread: MagicMock, mock_config: GlobalConfig) -> None:
    """
    Tests that the Orchestrator uses streaming (iread) instead of loading all data (read).
    """
    mock_config.max_cycles = 1
    explorer = MockExplorer(mock_config.explorer, mock_config.work_dir)
    oracle = MockOracle(mock_config.work_dir)
    trainer = MockTrainer(mock_config.trainer)
    validator = MockValidator(mock_config.validator)

    orchestrator = Orchestrator(mock_config, explorer, oracle, trainer, validator)

    # Set up mock_iread to yield something so the loop runs
    # It must yield atoms objects
    from ase import Atoms

    real_atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    mock_iread.return_value = [real_atoms]

    orchestrator.run()

    # Assert read was NOT called for loading labeled data
    # Note: orchestrator.run calls trainer.train which takes Dataset(file_path).
    # The read calls inside Trainer/Oracle are not patched here, only orchestrator.py imports.
    # In orchestrator.run:
    # ... labeled_data = oracle.label(...)
    # ... structures = read(labeled_data.file_path, index=":") <- This is what we want to avoid

    # We expect mock_read NOT to be called with labeled_data path
    # But wait, does mock_read catch `read` imported in orchestrator.py? Yes.

    # However, existing implementation calls `read`. So this test should FAIL until we fix it.
    mock_iread.assert_called()
