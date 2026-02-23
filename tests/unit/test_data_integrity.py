"""Tests for data integrity validation."""

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.core.utils import validate_structure_integrity, validate_structure_integrity_atoms
from pyacemaker.domain_models.models import StructureMetadata


def test_validate_atoms_integrity() -> None:
    # Valid atoms
    a = Atoms("H")
    validate_structure_integrity_atoms(a)

    # NaN positions
    a.positions[0] = [np.nan, 0, 0]
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_structure_integrity_atoms(a)

    # Inf positions
    a.positions[0] = [np.inf, 0, 0]
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_structure_integrity_atoms(a)

    # Periodic
    a = Atoms("H", cell=[1, 1, 1], pbc=True)
    validate_structure_integrity_atoms(a)

    # NaN cell
    a.cell[0] = [np.nan, 0, 0]
    with pytest.raises(ValueError, match="Structure cell contains NaN"):
        validate_structure_integrity_atoms(a)

    # Singular cell
    # Note: Atoms constructor handles cell, we modify it after or construct explicitly
    a = Atoms("H", pbc=True)
    a.set_cell([[0,0,0],[0,1,0],[0,0,1]])  # type: ignore[no-untyped-call] # volume 0
    with pytest.raises(ValueError, match="Structure cell volume is near zero"):
        validate_structure_integrity_atoms(a)

def test_validate_metadata_integrity() -> None:
    a = Atoms("H")
    s = StructureMetadata(features={"atoms": a})
    validate_structure_integrity(s)

    # Mismatched forces
    s.forces = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] # 2 forces
    # atoms has 1 atom
    with pytest.raises(ValueError, match="Forces array length"):
        validate_structure_integrity(s)
