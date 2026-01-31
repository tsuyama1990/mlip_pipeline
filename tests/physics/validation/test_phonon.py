from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np
from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig

def test_phonon_validator_pass():
    with patch("mlip_autopipec.physics.validation.phonon.Phonopy") as MockPhonopy, \
         patch("mlip_autopipec.physics.validation.phonon.get_reference_structure") as MockGetRef, \
         patch("mlip_autopipec.physics.validation.phonon.get_calculator"):

        from mlip_autopipec.physics.validation.phonon import PhononValidator

        # Mock atoms with valid cell
        mock_struct = MagicMock()
        mock_atoms = MagicMock()
        mock_atoms.cell = np.eye(3)
        mock_atoms.get_chemical_symbols.return_value = ["Si"]
        mock_atoms.get_scaled_positions.return_value = np.zeros((1,3))
        mock_struct.to_ase.return_value = mock_atoms
        MockGetRef.return_value = mock_struct

        mock_pho = MockPhonopy.return_value
        # Mock band structure
        freqs = [np.array([[1.0, 2.0], [1.5, 2.5]])]

        mock_pho.get_band_structure_dict.return_value = {
            'frequencies': freqs,
            'qpoints': [],
            'distances': []
        }

        # Mock supercells return
        mock_pho.supercells_with_displacements = []

        pot_config = PotentialConfig(elements=["Si"], cutoff=5.0)
        validator = PhononValidator(ValidationConfig(), pot_config)
        result = validator.validate(Path("pot.yace"))

        assert result.overall_status == "PASS"

def test_phonon_validator_fail():
    with patch("mlip_autopipec.physics.validation.phonon.Phonopy") as MockPhonopy, \
         patch("mlip_autopipec.physics.validation.phonon.get_reference_structure") as MockGetRef, \
         patch("mlip_autopipec.physics.validation.phonon.get_calculator"):

        from mlip_autopipec.physics.validation.phonon import PhononValidator

        mock_struct = MagicMock()
        mock_atoms = MagicMock()
        mock_atoms.cell = np.eye(3)
        mock_atoms.get_chemical_symbols.return_value = ["Si"]
        mock_atoms.get_scaled_positions.return_value = np.zeros((1,3))
        mock_struct.to_ase.return_value = mock_atoms
        MockGetRef.return_value = mock_struct

        mock_pho = MockPhonopy.return_value
        # Negative frequency < -0.1
        freqs = [np.array([[-0.2, 2.0], [1.5, 2.5]])]

        mock_pho.get_band_structure_dict.return_value = {
            'frequencies': freqs,
            'qpoints': [],
            'distances': []
        }
        mock_pho.supercells_with_displacements = []

        pot_config = PotentialConfig(elements=["Si"], cutoff=5.0)
        validator = PhononValidator(ValidationConfig(), pot_config)
        result = validator.validate(Path("pot.yace"))

        assert result.overall_status == "FAIL"
