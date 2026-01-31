from unittest.mock import MagicMock, patch

import numpy as np
from ase import Atoms

from mlip_autopipec.domain_models.config import ValidationConfig
from mlip_autopipec.physics.validation.phonon import PhononValidator


def test_phonon_validation_success(tmp_path):
    """Test that Phonon validation passes for stable band structure."""
    structure = Atoms(
        "Si2",
        positions=[[0, 0, 0], [1.5, 1.5, 1.5]],
        cell=np.eye(3) * 4,
        pbc=True,
    )
    calc = MagicMock()
    config = ValidationConfig(phonon_tolerance=-0.1)

    with patch("mlip_autopipec.physics.validation.phonon.Phonopy") as MockPhonopy:
        mock_instance = MockPhonopy.return_value

        # Mock supercells
        mock_sc = MagicMock()
        mock_sc.symbols = ["Si"]
        mock_sc.cell = np.eye(3) * 8
        mock_sc.positions = [[0, 0, 0]]
        mock_instance.supercells_with_displacements = [mock_sc]

        # Mock mesh
        mock_instance.get_mesh_dict.return_value = {
            "frequencies": np.array([1.0, 2.0, 3.0])
        }

        # Mock auto_band_structure
        mock_plot = MagicMock()
        mock_instance.auto_band_structure.return_value = mock_plot

        validator = PhononValidator(structure, calc, config, tmp_path, "test_pot")
        result = validator.validate()

        assert result.overall_status == "PASS"
        assert result.metrics[0].value == 1.0
        mock_plot.savefig.assert_called_once()


def test_phonon_validation_failure(tmp_path):
    """Test that Phonon validation fails for imaginary frequencies."""
    structure = Atoms(
        "Si2",
        positions=[[0, 0, 0], [1.5, 1.5, 1.5]],
        cell=np.eye(3) * 4,
        pbc=True,
    )
    calc = MagicMock()
    config = ValidationConfig(phonon_tolerance=-0.1)

    with patch("mlip_autopipec.physics.validation.phonon.Phonopy") as MockPhonopy:
        mock_instance = MockPhonopy.return_value

        # Mock supercells to avoid iteration error if code iterates
        mock_sc = MagicMock()
        mock_sc.symbols = ["Si"]
        mock_sc.cell = np.eye(3)
        mock_sc.positions = [[0, 0, 0]]
        mock_instance.supercells_with_displacements = [mock_sc]

        # Mock negative frequencies
        mock_instance.get_mesh_dict.return_value = {
            "frequencies": np.array([-1.0, 2.0, 3.0])
        }

        validator = PhononValidator(structure, calc, config, tmp_path, "test_pot")
        result = validator.validate()

        assert result.overall_status == "FAIL"
        assert result.metrics[0].value == -1.0
        assert result.metrics[0].passed is False
