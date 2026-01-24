from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from pydantic import TypeAdapter, ValidationError

from mlip_autopipec.data_models.atoms import ASEAtoms


def test_ase_atoms_valid():
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10])
    ta = TypeAdapter(ASEAtoms)
    validated = ta.validate_python(atoms)
    assert validated == atoms

def test_ase_atoms_invalid_type():
    ta = TypeAdapter(ASEAtoms)
    with pytest.raises(ValidationError):
        ta.validate_python("not an atoms object")

def test_ase_atoms_duck_typing():
    class MockAtoms:
        positions: ClassVar[np.ndarray] = np.array([[0,0,0]])
        cell: ClassVar[np.ndarray] = np.eye(3)
        numbers: ClassVar[list[int]] = [1]

        def __len__(self) -> int:
            return 1

    mock = MockAtoms()
    ta = TypeAdapter(ASEAtoms)
    validated = ta.validate_python(mock)
    assert validated == mock

def test_ase_atoms_shape_mismatch():
    # ASE Atoms prevents setting invalid shape, so we use a mock
    class BadAtoms:
        positions: ClassVar[np.ndarray] = np.array([[0, 0, 0], [1, 1, 1]])
        cell: ClassVar[np.ndarray] = np.eye(3)
        numbers: ClassVar[list[int]] = [1]
        def __len__(self) -> int: return 1

    ta = TypeAdapter(ASEAtoms)
    with pytest.raises((ValidationError, ValueError)) as exc:
        ta.validate_python(BadAtoms())
    assert "shape mismatch" in str(exc.value)
