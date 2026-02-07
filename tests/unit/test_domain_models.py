from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from pydantic import ValidationError

from mlip_autopipec.domain_models import (
    GlobalConfig,
    Potential,
    Structure,
    ValidationResult,
)


def test_structure_creation() -> None:
    atoms = Atoms(
        "H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], cell=[5, 5, 5], pbc=True
    )
    s = Structure(atoms=atoms)
    assert len(s.atoms) == 3
    assert s.energy is None
    assert s.forces is None

    # Test serialization
    dump = s.model_dump()
    assert dump["atoms"]["symbols"] == "H2O"
    assert len(dump["atoms"]["positions"]) == 3


def test_structure_validation_forces() -> None:
    atoms = Atoms("H")
    # Forces as list
    s = Structure(atoms=atoms, forces=[[0, 0, 0]])
    assert isinstance(s.forces, np.ndarray)
    assert s.forces.shape == (1, 3)


def test_config_validation() -> None:
    # Minimal valid config
    config_data: dict[str, object] = {
        "max_cycles": 10,
        "initial_structure_path": "data/start.xyz",
        "workdir": "work",
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "generator": {"type": "mock"},
        "validator": {"type": "mock"},
        "selector": {"type": "mock"},
    }
    config = GlobalConfig(**config_data)
    assert config.max_cycles == 10
    assert config.oracle.type == "mock"


def test_config_invalid() -> None:
    with pytest.raises(ValidationError):
        GlobalConfig(
            max_cycles=-1,
            initial_structure_path=Path(),
            workdir=Path(),
            oracle={"type": "mock"},
            trainer={"type": "mock"},
            dynamics={"type": "mock"},
            generator={"type": "mock"},
            validator={"type": "mock"},
            selector={"type": "mock"},
        )


def test_potential_model() -> None:
    p = Potential(path=Path("model.yace"))
    assert p.path == Path("model.yace")


def test_validation_result() -> None:
    res = ValidationResult(passed=True, metrics={"rmse": 0.1})
    assert res.passed
    assert res.metrics["rmse"] == 0.1

def test_structure_reconstruction_from_dict() -> None:
    data = {
        "symbols": "H2O",
        "positions": [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        "cell": [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
        "pbc": [True, True, True]
    }
    s = Structure(atoms=data)
    assert len(s.atoms) == 3
    assert str(s.atoms.symbols) == "H2O"
