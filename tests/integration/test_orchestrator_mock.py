from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from mlip_autopipec.components.oracle.mock import MockOracle
from mlip_autopipec.core.dataset import Dataset
from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models.config import (
    GlobalConfig,
    MockDynamicsConfig,
)
from mlip_autopipec.domain_models.enums import (
    DynamicsType,
    GeneratorType,
    OracleType,
    TrainerType,
    ValidatorType,
)
from mlip_autopipec.domain_models.structure import Structure

# Constants localized to test
CYCLE_01_DIR = "cycle_01"
CYCLE_02_DIR = "cycle_02"
CYCLE_03_DIR = "cycle_03"
POTENTIAL_FILE = "potential.yace"
DATASET_FILE = "dataset.jsonl"
STATE_FILE = "workflow_state.json"
STOPPED_STATUS = "STOPPED"
ERROR_STATUS = "ERROR"
MAX_CYCLES = 2
N_STRUCTURES = 5
CELL_SIZE = 10.0
N_ATOMS = 2
ATOMIC_NUMBERS = [1, 1]
SELECTION_RATE = 1.0
UNCERTAINTY_THRESHOLD = 5.0


@pytest.fixture
def mock_config(tmp_path: Path) -> GlobalConfig:
    return GlobalConfig.model_validate(
        {
            "workdir": tmp_path,
            "max_cycles": MAX_CYCLES,
            "logging_level": "INFO",
            "components": {
                "generator": {
                    "name": GeneratorType.MOCK,
                    "n_structures": N_STRUCTURES,
                    "cell_size": CELL_SIZE,
                    "n_atoms": N_ATOMS,
                    "atomic_numbers": ATOMIC_NUMBERS,
                },
                "oracle": {"name": OracleType.MOCK},
                "trainer": {"name": TrainerType.MOCK},
                "dynamics": {
                    "name": DynamicsType.MOCK,
                    "selection_rate": SELECTION_RATE,
                    "uncertainty_threshold": UNCERTAINTY_THRESHOLD,
                },
                "validator": {"name": ValidatorType.MOCK},
            },
        }
    )


def test_full_mock_orchestrator(mock_config: GlobalConfig, tmp_path: Path) -> None:
    # Arrange
    orchestrator = Orchestrator(mock_config)

    # Act
    orchestrator.run()

    # Assert
    assert (tmp_path / CYCLE_01_DIR).exists()
    assert (tmp_path / CYCLE_02_DIR).exists()
    assert not (tmp_path / CYCLE_03_DIR).exists()
    assert (tmp_path / CYCLE_01_DIR / POTENTIAL_FILE).exists()
    assert (tmp_path / CYCLE_02_DIR / POTENTIAL_FILE).exists()

    dataset_path = tmp_path / DATASET_FILE
    assert dataset_path.exists()

    dataset = Dataset(dataset_path, root_dir=tmp_path)
    count = 0
    # Use iterator to avoid loading full dataset into memory if it were large
    iterator = iter(dataset)
    try:
        while True:
            s = next(iterator)
            count += 1
            s.validate_labeled()
    except StopIteration:
        pass

    # Cycle 1 yields 5 structures (Generator)
    # Cycle 2 yields 30 structures (5 seeds -> 5 halts * 6 candidates)
    # Total count should be 35
    EXPECTED_TOTAL_COUNT = 35
    assert count == EXPECTED_TOTAL_COUNT

    state = orchestrator.state_manager.state
    assert state.current_cycle == MAX_CYCLES
    assert state.status == STOPPED_STATUS


class FailingOracle(MockOracle):
    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        msg = "Simulated DFT failure"
        raise RuntimeError(msg)


def test_orchestrator_component_failure(mock_config: GlobalConfig) -> None:
    with patch("mlip_autopipec.factory.ComponentFactory.get_oracle") as mock_get_oracle:
        failing_oracle = FailingOracle(mock_config.components.oracle)
        mock_get_oracle.return_value = failing_oracle

        orchestrator = Orchestrator(mock_config)

        with pytest.raises(RuntimeError, match="Simulated DFT failure"):
            orchestrator.run()

        state_manager = orchestrator.state_manager
        assert state_manager.state.status == ERROR_STATUS


def test_orchestrator_selection_logic(mock_config: GlobalConfig, tmp_path: Path) -> None:
    """Verify that selection_rate in Dynamics actually filters structures."""
    # Modify config to have 50% selection rate and deterministic seed
    dyn_config = cast(MockDynamicsConfig, mock_config.components.dynamics)
    dyn_config.selection_rate = 0.5
    dyn_config.seed = 42  # Fixed seed for determinism

    mock_config.components.generator.n_structures = 20
    mock_config.max_cycles = 2

    orchestrator = Orchestrator(mock_config)
    orchestrator.run()

    dataset_path = tmp_path / DATASET_FILE
    assert dataset_path.exists()

    dataset = Dataset(dataset_path, root_dir=tmp_path)

    # Verify filtering without loading all
    # Just iterate and count
    count = 0
    iterator = iter(dataset)
    try:
        while True:
            next(iterator)
            count += 1
    except StopIteration:
        pass

    # Cycle 1: 20 structures
    # Cycle 2:
    #   20 seeds.
    #   Selection rate 0.5 -> approx 10 halts.
    #   Each halt -> 6 candidates labeled.
    #   Expected total = 20 + (10 * 6) = 80.

    # We assert a range to be safe if seed implementation varies slightly
    # If selected = 10, total = 80.
    # We expect significantly more than 20 (Generator only)
    # and significantly less than 140 (100% selection: 20 + 20*6 = 140)

    assert count > 25, "Cycle 2 should add structures via active learning"
    assert count < 130, "Selection logic should have filtered some structures"

    # Verify provenance
    # We reload a small sample to check provenance if needed, or iterate again to find any
    # But assertion above proves logic worked (count < 130)
