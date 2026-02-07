import numpy as np
import pytest

from mlip_autopipec.domain_models import Structure
from mlip_autopipec.infrastructure.generator import RandomDisplacement


def test_random_displacement_init() -> None:
    gen = RandomDisplacement(params={"magnitude": 0.1})
    assert gen.params["magnitude"] == 0.1

def test_random_displacement_generate_returns_copy() -> None:
    # Test that it returns a copy and modifies positions
    positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    cell = np.eye(3) * 10.0
    species = ["Ar", "Ar"]
    base_structure = Structure(positions=positions, cell=cell, species=species)

    gen = RandomDisplacement(params={"magnitude": 0.1})

    candidates = gen.generate(base_structure)
    assert len(candidates) == 1
    new_struct = candidates[0]
    assert new_struct is not base_structure

def test_random_displacement_logic() -> None:
    positions = np.array([[0.0, 0.0, 0.0]])
    cell = np.eye(3) * 10.0
    species = ["Ar"]
    base_structure = Structure(positions=positions, cell=cell, species=species)

    magnitude = 0.1
    gen = RandomDisplacement(params={"magnitude": magnitude})

    candidates = gen.generate(base_structure)
    assert len(candidates) > 0
    new_struct = candidates[0]

    # Check it's a copy
    assert new_struct is not base_structure

    # Check positions moved
    # Note: We check that at least one coordinate changed
    diff = new_struct.positions - base_structure.positions
    assert np.any(np.abs(diff) > 0)

    # Check max displacement (assuming uniform distribution [-mag, mag])
    assert np.all(np.abs(diff) <= magnitude)
