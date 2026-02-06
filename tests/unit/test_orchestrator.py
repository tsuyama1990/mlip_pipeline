from pathlib import Path
from unittest.mock import MagicMock

import pytest
from ase.io import read

from mlip_autopipec.config import (
    ExplorerConfig,
    GlobalConfig,
    OracleConfig,
    TrainerConfig,
    ValidatorConfig,
)
from mlip_autopipec.constants import DEFAULT_POTENTIAL_NAME
from mlip_autopipec.domain_models import Dataset, ValidationResult
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
    assert orchestrator.current_potential_path == Path(DEFAULT_POTENTIAL_NAME)
    # Check dataset file path setup
    assert orchestrator.dataset_file == mock_config.work_dir / "accumulated_dataset.xyz"
    assert orchestrator.dataset_file.exists()


def test_orchestrator_run_mocked_interfaces(mock_config: GlobalConfig) -> None:
    """
    Tests the orchestration logic using mocked interfaces to verify call counts and flow.
    """
    explorer = MagicMock()
    oracle = MagicMock()
    trainer = MagicMock()
    validator = MagicMock()

    # Orchestrator
    orchestrator = Orchestrator(mock_config, explorer, oracle, trainer, validator)

    # Setup return values
    candidates_file = mock_config.work_dir / "candidates.xyz"
    candidates_file.touch() # Create empty file for candidates (Orchestrator just logs path)

    labeled_file = mock_config.work_dir / "labeled.xyz"
    # Create valid labeled file content for ASE iread
    # 2 atoms of H
    with labeled_file.open("w") as f:
        f.write("2\nProperties=species:S:1:pos:R:3 energy=-10.0\nH 0.0 0.0 0.0\nH 1.0 0.0 0.0\n")

    explorer.explore.return_value = Dataset(file_path=candidates_file)
    oracle.label.return_value = Dataset(file_path=labeled_file)
    trainer.train.return_value = mock_config.work_dir / "potential.yace"
    validator.validate.return_value = ValidationResult(metrics={"rmse": 0.1}, is_stable=True)

    # Run
    orchestrator.run()

    # Assertions
    assert explorer.explore.call_count == 2
    assert oracle.label.call_count == 2
    # Orchestrator calls label(new_candidates)

    # Check dataset accumulation
    dataset_file = mock_config.work_dir / "accumulated_dataset.xyz"
    assert dataset_file.exists()
    # Should have appended 2 structures twice (one per cycle) -> 4 structures?
    # Wait, labeled_file has 1 frame with 2 atoms. "2\n..." is one frame.
    # So 1 frame per cycle -> 2 frames total.
    frames = read(dataset_file, index=":")
    assert len(frames) == 2


def test_orchestrator_run_integration(mock_config: GlobalConfig) -> None:
    """
    Tests the full execution cycle of the Orchestrator using the real Mock components.
    This verifies that the Mock components interact correctly with the Orchestrator.
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

    # MockExplorer generates n_structures=2 per call.
    # MockOracle labels them.
    # 2 cycles -> 4 structures total.
    structures = read(dataset_file, index=":")
    if not isinstance(structures, list):
        structures = [structures]
    assert len(structures) == 4

    # Check potential path updated.
    expected_potential = mock_config.work_dir / "potential.yace"
    assert orchestrator.current_potential_path == expected_potential
    assert expected_potential.exists()
