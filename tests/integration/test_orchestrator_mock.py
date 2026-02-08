from pathlib import Path
from typing import Any, cast
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
from tests.common_constants import (
    ATOMIC_NUMBERS,
    CELL_SIZE,
    CYCLE_01_DIR,
    CYCLE_02_DIR,
    CYCLE_03_DIR,
    DATASET_FILE,
    ERROR_STATUS,
    MAX_CYCLES,
    N_ATOMS,
    N_STRUCTURES,
    POTENTIAL_FILE,
    SELECTION_RATE,
    STATE_FILE,
    STOPPED_STATUS,
    UNCERTAINTY_THRESHOLD,
)


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
                    # Explicitly required params
                    "cell_size": CELL_SIZE,
                    "n_atoms": N_ATOMS,
                    "atomic_numbers": ATOMIC_NUMBERS,
                },
                "oracle": {"name": OracleType.MOCK},
                "trainer": {"name": TrainerType.MOCK},
                "dynamics": {
                    "name": DynamicsType.MOCK,
                    "selection_rate": SELECTION_RATE,
                    "uncertainty_threshold": UNCERTAINTY_THRESHOLD,  # Required for mock now
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
    # Check cycles
    assert (tmp_path / CYCLE_01_DIR).exists()
    assert (tmp_path / CYCLE_02_DIR).exists()
    assert not (tmp_path / CYCLE_03_DIR).exists()

    # Check potential files
    assert (tmp_path / CYCLE_01_DIR / POTENTIAL_FILE).exists()
    assert (tmp_path / CYCLE_02_DIR / POTENTIAL_FILE).exists()

    # Check dataset
    dataset_path = tmp_path / DATASET_FILE
    assert dataset_path.exists()

    # Verify data flow: check if structures have been processed
    # Use streaming iterator to avoid loading full list
    dataset = Dataset(dataset_path)
    count = 0

    # Streaming iteration: do not use list(dataset) to ensure scalability
    # Dataset implements __iter__ which yields structures one by one
    iterator = iter(dataset)
    try:
        while True:
            s = next(iterator)
            count += 1
            # Verify labeling happened
            assert s.energy is not None
            assert s.forces is not None
            assert s.stress is not None
            # Verify integrity
            s.validate_labeled()
    except StopIteration:
        pass

    # Cycle 1: 5 structures (selection_rate=1.0)
    # Cycle 2: 5 structures selected from generated
    # Total 10.
    EXPECTED_TOTAL_COUNT = 10
    assert count == EXPECTED_TOTAL_COUNT

    # Check state
    assert (tmp_path / STATE_FILE).exists()
    state = orchestrator.state_manager.state
    assert state.current_cycle == MAX_CYCLES
    assert state.status == STOPPED_STATUS


class FailingOracle(MockOracle):
    def compute(self, structures: Any) -> Any:
        msg = "Simulated DFT failure"
        raise RuntimeError(msg)


def test_orchestrator_component_failure(mock_config: GlobalConfig) -> None:
    # Use dependency injection by patching the factory method to return the failing component
    # This avoids setting internal attributes of Orchestrator directly.
    # We patch ComponentFactory.get_oracle to return our FailingOracle instance.

    # We create the config inside the FailingOracle constructor or pass it mocked.
    # Actually Orchestrator calls get_oracle with config from config file.
    # We can just return the FailingOracle regardless of config passed to get_oracle.

    with patch("mlip_autopipec.factory.ComponentFactory.get_oracle") as mock_get_oracle:
        # Configure mock to return FailingOracle
        # FailingOracle needs a config to init base class
        # But we mock the return value of factory, so we instantiate it here.
        # We need to ensure config type is correct for MockOracle (base).
        # We can construct it with mock_config's oracle config which is MockOracleConfig.

        failing_oracle = FailingOracle(mock_config.components.oracle) # type: ignore
        mock_get_oracle.return_value = failing_oracle

        orchestrator = Orchestrator(mock_config)

        # Verify graceful failure
        with pytest.raises(RuntimeError, match="Simulated DFT failure"):
            orchestrator.run()

        # State should be updated to ERROR
        # Need to reload state from file to verify persistence
        state_manager = orchestrator.state_manager
        # We can check the in-memory state object as it should be updated
        assert state_manager.state.status == ERROR_STATUS

        # Also verify file persistence
        # Re-instantiate StateManager to read from file
        from mlip_autopipec.core.state import StateManager

        loaded_state = StateManager(mock_config.workdir / STATE_FILE).state
        assert loaded_state.status == ERROR_STATUS


def test_orchestrator_selection_logic(mock_config: GlobalConfig, tmp_path: Path) -> None:
    """Verify that selection_rate in Dynamics actually filters structures."""
    # Modify config to have 50% selection rate and deterministic seed
    dyn_config = cast(MockDynamicsConfig, mock_config.components.dynamics)
    dyn_config.selection_rate = 0.5
    dyn_config.seed = 42  # Ensure MockDynamics uses this seed for its local RNG

    # Generate enough structures to be statistically significant or just deterministic
    mock_config.components.generator.n_structures = 20
    # Cycle 1 is cold start (labels all 20). Cycle 2 uses dynamics (filters ~50% of 20 -> ~10).
    mock_config.max_cycles = 2

    orchestrator = Orchestrator(mock_config)
    orchestrator.run()

    # Check dataset count
    dataset_path = tmp_path / DATASET_FILE
    assert dataset_path.exists()

    dataset = Dataset(dataset_path)
    count = 0
    # Scalability: Use manual iteration to avoid any potential full materialization
    # even though sum() on generator is theoretically fine, explicit loop is safer.
    iterator = iter(dataset)
    try:
        while True:
            next(iterator)
            count += 1
    except StopIteration:
        pass

    # Cycle 1: 20
    # Cycle 2: ~10 (with seed 42, likely 11 based on check_rng.py)
    # Total ~30.
    # We verify it's significantly less than 40 (which would be no filtering)
    # and significantly more than 20 (which would be no cycle 2).
    # Expected value: 20 (cycle 1) + 20 * 0.5 (cycle 2) = 30.
    # Allowing some variance due to randomness.
    # 20 + 11 = 31 (with seed 42).
    EXPECTED_MIN = 25
    EXPECTED_MAX = 35
    assert EXPECTED_MIN <= count <= EXPECTED_MAX
    # This proves dynamics filtered roughly half.
