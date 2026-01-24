from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.validation import PhononConfig
from mlip_autopipec.validation.phonon import PhononValidator


@pytest.fixture
def mock_phonopy():
    with patch("mlip_autopipec.validation.phonon.Phonopy") as MockPhonopy:
        yield MockPhonopy


def test_phonon_validator_initialization(mock_phonopy):
    config = PhononConfig()
    validator = PhononValidator(config)
    assert validator.config == config


def test_phonon_validator_stable(mock_phonopy):
    config = PhononConfig()
    validator = PhononValidator(config)
    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
    calculator = MagicMock()

    # Mock return values
    instance = mock_phonopy.return_value
    # Mock supercells
    sc = Atoms("Al2", positions=[[0, 0, 0], [2, 2, 2]], cell=[8, 8, 8])
    instance.supercell = sc
    instance.supercells_with_displacements = [sc, sc]

    # Mock frequencies (positive)
    # run_mesh() sets mesh properties in object
    # get_mesh_dict() returns dict with 'frequencies'
    instance.get_mesh_dict.return_value = {"frequencies": np.array([[0.1, 2.0], [3.0, 4.0]])}

    result = validator.validate(atoms, calculator)
    assert result is True

    # Verify calculator calls
    assert calculator.get_forces.call_count >= 1


def test_phonon_validator_unstable(mock_phonopy):
    config = PhononConfig()
    validator = PhononValidator(config)
    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
    calculator = MagicMock()

    instance = mock_phonopy.return_value
    sc = Atoms("Al2", positions=[[0, 0, 0], [2, 2, 2]], cell=[8, 8, 8])
    instance.supercell = sc
    instance.supercells_with_displacements = [sc]

    # Mock frequencies (negative)
    # Typically small negative values are allowed (tolerance), but large ones are instability
    instance.get_mesh_dict.return_value = {"frequencies": np.array([[-5.0, 2.0], [3.0, 4.0]])}

    result = validator.validate(atoms, calculator)
    assert result is False
