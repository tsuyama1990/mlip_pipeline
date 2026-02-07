from pathlib import Path

import numpy as np
import pytest

from mlip_autopipec.domain_models import (
    MockDynamicsConfig,
    MockOracleConfig,
    MockTrainerConfig,
    Structure,
)
from mlip_autopipec.infrastructure.mocks import MockDynamics, MockOracle, MockTrainer


def test_scenario_02(tmp_path: Path) -> None:
    """
    SCENARIO 02: Mock Execution Loop
    Verify that the Mock components interact correctly through the defined interfaces.
    """
    workdir = tmp_path / "work"
    workdir.mkdir()

    # Instantiate Mocks
    oracle = MockOracle(MockOracleConfig(noise_level=0.1))
    trainer = MockTrainer(MockTrainerConfig())
    dynamics = MockDynamics(MockDynamicsConfig())

    # 1. Oracle Compute
    structures = [
        Structure(
            positions=np.zeros((2, 3)),
            atomic_numbers=np.array([1, 1]),
            cell=np.eye(3),
            pbc=np.array([True, True, True]),
        )
    ]
    labeled_structures = list(oracle.compute(structures))
    assert len(labeled_structures) == 1
    assert labeled_structures[0].energy is not None
    assert labeled_structures[0].forces is not None

    # 2. Trainer Train
    potential = trainer.train(labeled_structures, workdir)
    assert potential.path.exists()
    assert (workdir / "dummy.yace").exists()

    # 3. Dynamics Run
    result = dynamics.run(potential, structures, workdir)
    assert result.status in ["converged", "active", "failed"]

    # If active, we expect new structures
    if result.status == "active":
        assert len(result.structures) > 0
        assert isinstance(result.structures[0], Structure)
