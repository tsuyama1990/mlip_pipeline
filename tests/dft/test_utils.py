"""
Tests for DFT utilities.
"""
import pytest
from ase import Atoms
from mlip_autopipec.dft import utils

def test_get_kpoints():
    """Test K-point grid generation."""
    # 10x10x10 cell (large) -> Small reciprocal cell
    atoms_large = Atoms(cell=[10, 10, 10], pbc=True)
    # |b| ~ 2pi/10 = 0.628
    # If density=0.15, k = 0.628 * 0.15 = 0.09 -> 1
    kpts_large = utils.get_kpoints(atoms_large, density=0.15)
    assert kpts_large == [1, 1, 1]

    # 2x2x2 cell (small) -> Large reciprocal cell
    atoms_small = Atoms(cell=[2, 2, 2], pbc=True)
    # |b| ~ 2pi/2 = 3.14159
    # Spec expectation: [5, 5, 5]
    # To get 5: 5 = 3.14159 * density => density ~ 1.59
    kpts_small = utils.get_kpoints(atoms_small, density=1.6)
    assert kpts_small == [5, 5, 5]

def test_is_magnetic():
    """Test magnetism detection."""
    assert utils.is_magnetic(Atoms("Fe"))
    assert utils.is_magnetic(Atoms("Ni"))
    assert utils.is_magnetic(Atoms("Co"))
    assert not utils.is_magnetic(Atoms("Si"))
    assert not utils.is_magnetic(Atoms("Al"))
    assert utils.is_magnetic(Atoms("FeSi"))

def test_get_sssp_pseudopotentials():
    """Test pseudopotential mapping."""
    elements = ["Fe", "Si"]
    pseudos = utils.get_sssp_pseudopotentials(elements)
    assert "Fe" in pseudos
    assert "Si" in pseudos
    assert pseudos["Fe"].endswith(".UPF")
