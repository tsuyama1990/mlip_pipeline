from pathlib import Path

import numpy as np
import pytest

from mlip_autopipec.components.dynamics import MockDynamics
from mlip_autopipec.components.generator import MockGenerator
from mlip_autopipec.components.oracle import MockOracle
from mlip_autopipec.components.trainer import MockTrainer
from mlip_autopipec.components.validator import MockValidator
from mlip_autopipec.domain_models import Potential, Structure


def test_mock_generator_fail() -> None:
    gen = MockGenerator(fail_rate=1.0)
    with pytest.raises(RuntimeError, match="Simulated failure"):
        list(gen.generate())


def test_mock_oracle_fail() -> None:
    oracle = MockOracle(fail_rate=1.0)
    s = Structure(
        positions=np.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=np.array([1]),
        cell=np.eye(3),
    )
    with pytest.raises(RuntimeError, match="Simulated failure"):
        list(oracle.compute([s]))


def test_mock_trainer_fail() -> None:
    trainer = MockTrainer(fail_rate=1.0)
    with pytest.raises(RuntimeError, match="Simulated failure"):
        trainer.train([], workdir=None)


def test_mock_dynamics_fail() -> None:
    dyn = MockDynamics(fail_rate=1.0)
    pot = Potential(path=Path("dummy"), version="v1")
    with pytest.raises(RuntimeError, match="Simulated failure"):
        list(dyn.run(pot))


def test_mock_validator_fail() -> None:
    val = MockValidator(fail_rate=1.0)
    pot = Potential(path=Path("dummy"), version="v1")
    with pytest.raises(RuntimeError, match="Simulated failure"):
        val.validate(pot)
