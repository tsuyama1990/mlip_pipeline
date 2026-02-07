from pathlib import Path

import numpy as np

from mlip_autopipec.domain_models import (
    MockDynamicsConfig,
    MockOracleConfig,
    MockTrainerConfig,
    Potential,
    Structure,
)

# Mock classes will be implemented in Step 6
from mlip_autopipec.infrastructure.mocks import MockDynamics, MockOracle, MockTrainer


def test_mock_oracle() -> None:
    """Test MockOracle functionality."""
    config = MockOracleConfig(noise_level=0.1)
    oracle = MockOracle(config)

    structures = [
        Structure(
            positions=np.zeros((2, 3)),
            atomic_numbers=np.array([1, 1]),
            cell=np.eye(3),
            pbc=np.array([True, True, True]),
        )
    ]

    results = list(oracle.compute(structures))
    assert len(results) == 1
    assert results[0].energy is not None
    assert results[0].forces is not None
    assert results[0].stress is not None


def test_mock_trainer(tmp_path: Path) -> None:
    """Test MockTrainer functionality."""
    config = MockTrainerConfig()
    trainer = MockTrainer(config)

    structures = [
        Structure(
            positions=np.zeros((2, 3)),
            atomic_numbers=np.array([1, 1]),
            cell=np.eye(3),
            pbc=np.array([True, True, True]),
        )
    ]

    potential = trainer.train(structures, tmp_path)
    assert isinstance(potential, Potential)
    assert potential.path.exists()
    assert potential.format == "yace"


def test_mock_dynamics(tmp_path: Path) -> None:
    """Test MockDynamics functionality."""
    config = MockDynamicsConfig()
    dynamics = MockDynamics(config)

    structures = [
        Structure(
            positions=np.zeros((2, 3)),
            atomic_numbers=np.array([1, 1]),
            cell=np.eye(3),
            pbc=np.array([True, True, True]),
        )
    ]

    # Create dummy potential file
    pot_path = tmp_path / "dummy.yace"
    pot_path.touch()
    potential = Potential(path=pot_path)

    result = dynamics.run(potential, structures, tmp_path)
    assert result.status in ["converged", "active", "failed"]
    assert isinstance(result.structures, list)
    # MockDynamics should generate new structures or return empty list
    if result.status == "active":
        assert len(result.structures) > 0
