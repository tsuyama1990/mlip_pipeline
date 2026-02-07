from pathlib import Path

import numpy as np
import pytest

from mlip_autopipec.domain_models import Potential, Structure
from mlip_autopipec.infrastructure import (
    MockDynamics,
    MockOracle,
    MockStructureGenerator,
    MockTrainer,
)


def test_scenario_1_2_mock_loop(tmp_path: Path) -> None:
    # 1. Initialize components
    oracle = MockOracle()
    trainer = MockTrainer()
    dynamics = MockDynamics(params={"halt_probability": 0.5})
    generator = MockStructureGenerator(params={"n_candidates": 2})

    # 2. Pass dummy structure to Dynamics
    initial_structure = Structure(
        symbols=["H", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        cell=np.eye(3),
        pbc=np.array([True, True, True]),
    )

    # Dummy potential needed for Dynamics run
    dummy_yace = tmp_path / "dummy.yace"
    dummy_yace.touch()
    potential = Potential(path=dummy_yace, version="v0")

    result = dynamics.run(potential, initial_structure)
    assert result.status in ["halted", "converged"]
    assert result.structure is not None

    # 3. If halted (or forced), generate candidates
    candidates = generator.generate(result.structure)
    assert len(candidates) == 2

    # 4. Pass candidates to Oracle
    labeled = oracle.compute(candidates)
    assert len(labeled) == 2
    assert "energy" in labeled[0].properties
    assert "forces" in labeled[0].properties

    # 5. Pass labeled to Trainer
    workdir = tmp_path / "training_run"
    new_potential = trainer.train(labeled, workdir)

    assert new_potential.path.exists()
    assert new_potential.path.name == "dummy.yace"
    assert new_potential.metrics["rmse_energy"] > 0.0
