import numpy as np
from ase import Atoms

from mlip_autopipec.training.physics import ZBLCalculator


def test_zbl_calculator_energy():
    calc = ZBLCalculator()
    # H-H at 0.5 Angstrom
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.5]])
    atoms.calc = calc
    energy = atoms.get_potential_energy()

    # Check it is positive (repulsive) and large
    assert energy > 0
    # Approximate check:
    # Z=1. a = 0.468 / (1+1) approx 0.23? No.
    # a = 0.8854*0.529 / (1^0.23 + 1^0.23) = 0.468 / 2 = 0.234
    # r/a = 0.5 / 0.234 = 2.13
    # phi(2.13) will be small but non-zero.
    # Coulomb = 14.4 / 0.5 = 28.8 eV
    # V = 28.8 * phi
    assert energy > 1.0 # Should be significant

def test_zbl_calculator_forces():
    calc = ZBLCalculator()
    # H-H at 0.5 Angstrom
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.5]])
    atoms.calc = calc
    forces = atoms.get_forces()

    # Force on atom 1 (at 0.5) should be along +z (repulsive)
    # Force on atom 0 (at 0.0) should be along -z

    assert forces[1][2] > 0
    assert forces[0][2] < 0
    assert np.allclose(forces[0], -forces[1])

def test_zbl_calculator_cutoff_behavior():
    calc = ZBLCalculator()
    # H-H at large distance
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 10.0]])
    atoms.calc = calc
    energy = atoms.get_potential_energy()

    # Should be effectively zero
    assert energy < 1e-3
