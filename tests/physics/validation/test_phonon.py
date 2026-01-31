from unittest.mock import MagicMock, patch
import pytest
from ase import Atoms

from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig
from mlip_autopipec.domain_models.validation import ValidationResult
from mlip_autopipec.physics.validation.phonon import PhononValidator

@pytest.fixture
def mock_atoms():
    return Atoms("Al", positions=[[0, 0, 0]], cell=[[4.05, 0, 0], [0, 4.05, 0], [0, 0, 4.05]], pbc=True)

@pytest.fixture
def validation_config():
    return ValidationConfig(phonon_tolerance=-0.1)

@pytest.fixture
def potential_config():
    return PotentialConfig(elements=["Al"], cutoff=5.0)

def test_phonon_validator_pass(mock_atoms, validation_config, potential_config, tmp_path):
    with patch("mlip_autopipec.physics.validation.phonon.Phonopy") as MockPhonopy:
        # Setup Mock Phonopy instance
        mock_instance = MockPhonopy.return_value

        # Mock produce_force_constants
        mock_instance.produce_force_constants.return_value = None

        # Mock get_band_structure_dict to return stable frequencies
        # Structure of dict: {'qpoints': [...], 'distances': [...], 'frequencies': [[freq1, freq2...], ...]}
        mock_instance.get_band_structure_dict.return_value = {
            'frequencies': [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]] # All positive
        }

        validator = PhononValidator(
            potential_path=tmp_path / "pot.yace",
            config=validation_config,
            potential_config=potential_config
        )

        # Mock calculator injection
        validator._get_calculator = MagicMock()

        result = validator.validate(reference_structure=mock_atoms)

        assert isinstance(result, ValidationResult)
        assert result.overall_status == "PASS"
        assert result.metrics[0].name == "Min Frequency"
        assert result.metrics[0].passed is True

def test_phonon_validator_fail(mock_atoms, validation_config, potential_config, tmp_path):
    with patch("mlip_autopipec.physics.validation.phonon.Phonopy") as MockPhonopy:
        mock_instance = MockPhonopy.return_value
        # Return unstable frequencies
        mock_instance.get_band_structure_dict.return_value = {
            'frequencies': [[-5.0, 2.0, 3.0]]
        }

        validator = PhononValidator(
            potential_path=tmp_path / "pot.yace",
            config=validation_config,
            potential_config=potential_config
        )
        validator._get_calculator = MagicMock()

        result = validator.validate(reference_structure=mock_atoms)

        assert result.overall_status == "FAIL"
        assert result.metrics[0].passed is False
