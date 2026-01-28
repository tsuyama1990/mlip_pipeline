from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from pydantic import TypeAdapter, ValidationError

from mlip_autopipec.domain_models.atoms import ASEAtoms


def test_ase_atoms_valid():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10])
    ta = TypeAdapter(ASEAtoms)
    validated = ta.validate_python(atoms)
    assert validated == atoms


def test_ase_atoms_invalid_type():
    ta = TypeAdapter(ASEAtoms)
    # Pydantic might wrap TypeError in ValidationError or let it bubble depending on context
    with pytest.raises((ValidationError, TypeError)):
        ta.validate_python("not an atoms object")


def test_ase_atoms_duck_typing():
    # Duck typing is explicitly disallowed now
    class MockAtoms:
        positions: ClassVar[np.ndarray] = np.array([[0, 0, 0]])
        cell: ClassVar[np.ndarray] = np.eye(3)
        numbers: ClassVar[list[int]] = [1]
        pbc: ClassVar[np.ndarray] = np.array([True, True, True])

        def __len__(self) -> int:
            return 1

        def get_positions(self):
            return self.positions

        def get_cell(self):
            return self.cell

        def get_atomic_numbers(self):
            return self.numbers

        def get_pbc(self):
            return self.pbc

    mock = MockAtoms()
    ta = TypeAdapter(ASEAtoms)

    with pytest.raises((ValidationError, TypeError), match="Expected ase.Atoms object"):
        ta.validate_python(mock)


def test_ase_atoms_shape_mismatch():
    class BadAtoms(Atoms):
        def __init__(self):
            pass

        @property
        def positions(self):
            return np.array([[0, 0, 0], [1, 1, 1]])  # 2 atoms

        @property
        def cell(self):
            return np.eye(3)

        @property
        def numbers(self):
            return [1]  # 1 atom - mismatch

        def __len__(self) -> int:
            return 1

        def get_positions(self):
            return self.positions

        def get_cell(self):
            return self.cell

        def get_atomic_numbers(self):
            return self.numbers

        def get_pbc(self):
            return [True] * 3

    ta = TypeAdapter(ASEAtoms)
    # The validator raises ValueError wrapped in ValidationError
    with pytest.raises((ValidationError, ValueError)) as exc:
        ta.validate_python(BadAtoms())

    # Check for our new error message
    assert "Positions count" in str(exc.value) or "does not match" in str(exc.value)
