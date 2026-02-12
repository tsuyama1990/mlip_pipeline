import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.domain_models.datastructures import Structure


@pytest.fixture
def valid_structure() -> Structure:
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    return Structure(atoms=atoms, provenance="test", forces=np.zeros((3, 3)))
