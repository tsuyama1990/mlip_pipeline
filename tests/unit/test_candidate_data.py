import pytest
from pydantic import ValidationError
from ase import Atoms
from mlip_autopipec.data_models.candidate import CandidateData
from unittest.mock import MagicMock
import numpy as np

def test_candidate_data_valid_atoms():
    # Valid atoms
    atoms = Atoms('H2', positions=[[0,0,0], [0,0,0.8]])
    data = CandidateData(atoms=atoms, config_type="test")
    assert data.atoms == atoms
    assert data.status == "pending"

def test_candidate_data_invalid_type():
    # Invalid type
    with pytest.raises(ValidationError):
        CandidateData(atoms="not an atoms object", config_type="test")

def test_candidate_data_malformed_atoms():
    # Use a Mock to simulate a corrupted ASE object that ASE itself might strictly prevent
    # but could theoretically happen via deep hacking or subclassing.
    mock_atoms = MagicMock(spec=Atoms)
    # Mock isinstance to return True? No, spec handles that for some checks, but isinstance might fail
    # if not actually inheriting.
    # But validate_ase_atoms checks isinstance(v, Atoms).
    # If I use a real Atoms object and force it?

    # Let's try subclassing
    class CorruptedAtoms(Atoms):
        def __len__(self):
            return 2
        @property
        def positions(self):
            return np.array([[0,0,0]]) # Shape (1,3)

    atoms = CorruptedAtoms('H2')

    # The ASEAtoms validator should catch this
    with pytest.raises(ValidationError) as excinfo:
        CandidateData(atoms=atoms, config_type="test")

    assert "Malformed Atoms object" in str(excinfo.value)

def test_candidate_data_extra_forbid():
    atoms = Atoms('H')
    with pytest.raises(ValidationError):
        CandidateData(atoms=atoms, config_type="test", extra_field="bad")
