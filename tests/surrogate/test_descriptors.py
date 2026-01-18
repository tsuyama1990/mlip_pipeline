import pytest
import numpy as np
from ase import Atoms
from mlip_autopipec.surrogate.descriptors import DescriptorCalculator

def test_descriptor_calculator_initialization():
    calc = DescriptorCalculator()
    assert calc is not None

def test_descriptor_shape():
    calc = DescriptorCalculator()
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    batch = [atoms] * 5
    descriptors = calc.compute_soap(batch)
    assert isinstance(descriptors, np.ndarray)
    assert descriptors.shape[0] == 5
    assert descriptors.shape[1] > 0
    # verify dtypes
    assert descriptors.dtype == np.float64 or descriptors.dtype == np.float32

def test_descriptor_rotational_invariance():
    calc = DescriptorCalculator()
    # H2 molecule
    atoms1 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.0]])
    # Rotated 90 degrees around Y axis
    atoms2 = Atoms('H2', positions=[[0, 0, 0], [1.0, 0, 0]])

    desc1 = calc.compute_soap([atoms1])[0]
    desc2 = calc.compute_soap([atoms2])[0]

    # Distance should be very small
    dist = np.linalg.norm(desc1 - desc2)
    assert dist < 1e-4

def test_descriptor_different_molecules():
    calc = DescriptorCalculator()
    atoms1 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.0]])
    atoms2 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 2.0]])

    desc1 = calc.compute_soap([atoms1])[0]
    desc2 = calc.compute_soap([atoms2])[0]

    dist = np.linalg.norm(desc1 - desc2)
    assert dist > 0.01
