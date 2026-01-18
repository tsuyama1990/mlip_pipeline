import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.surrogate import DescriptorConfig
from mlip_autopipec.surrogate.descriptors import DescriptorCalculator


@pytest.fixture
def default_config():
    return DescriptorConfig()


def test_descriptor_calculator_initialization(default_config):
    calc = DescriptorCalculator(default_config)
    assert calc is not None
    assert calc.config.r_cut == 6.0


def test_descriptor_shape(default_config):
    calc = DescriptorCalculator(default_config)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    batch = [atoms] * 5
    result = calc.compute_soap(batch)
    descriptors = result.features

    assert isinstance(descriptors, np.ndarray)
    assert descriptors.shape[0] == 5
    assert descriptors.shape[1] > 0
    # verify dtypes
    assert descriptors.dtype == np.float64 or descriptors.dtype == np.float32


def test_descriptor_rotational_invariance(default_config):
    calc = DescriptorCalculator(default_config)
    # H2 molecule
    atoms1 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.0]])
    # Rotated 90 degrees around Y axis
    atoms2 = Atoms("H2", positions=[[0, 0, 0], [1.0, 0, 0]])

    # Compute individually
    res1 = calc.compute_soap([atoms1])
    res2 = calc.compute_soap([atoms2])

    desc1 = res1.features[0]
    desc2 = res2.features[0]

    # Distance should be very small
    dist = np.linalg.norm(desc1 - desc2)
    assert dist < 1e-4


def test_descriptor_different_molecules(default_config):
    calc = DescriptorCalculator(default_config)
    atoms1 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.0]])
    atoms2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 2.0]])

    res1 = calc.compute_soap([atoms1])
    res2 = calc.compute_soap([atoms2])

    desc1 = res1.features[0]
    desc2 = res2.features[0]

    dist = np.linalg.norm(desc1 - desc2)
    assert dist > 0.01
