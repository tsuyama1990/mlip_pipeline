from unittest.mock import MagicMock, patch

import pytest
import numpy as np
from mlip_autopipec.domain_models.config import ValidationConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.validation.phonon import PhononValidator


@pytest.fixture
def mock_validation_config():
    return ValidationConfig()


@pytest.fixture
def mock_structure():
    return Structure(
        symbols=["Si"],
        positions=np.array([[0, 0, 0]]),
        cell=np.eye(3),
        pbc=(True, True, True),
    )


def test_phonon_validator_pass(mock_validation_config, mock_structure):
    with patch("mlip_autopipec.physics.validation.phonon.Phonopy") as mock_phonopy_cls:
        mock_phonopy = MagicMock()
        mock_phonopy_cls.return_value = mock_phonopy

        # Mock mesh
        # Stable: all > -tolerance
        freqs = np.array([1.0, 2.0, 3.0, 4.0])
        mock_phonopy.get_mesh_dict.return_value = {"frequencies": freqs}

        validator = PhononValidator(mock_validation_config, "pot.yace")
        with patch("mlip_autopipec.physics.validation.phonon.get_lammps_calculator"):
            result = validator.validate(mock_structure)

        assert result.overall_status == "PASS"


def test_phonon_validator_fail(mock_validation_config, mock_structure):
    with patch("mlip_autopipec.physics.validation.phonon.Phonopy") as mock_phonopy_cls:
        mock_phonopy = MagicMock()
        mock_phonopy_cls.return_value = mock_phonopy

        # Unstable: some < -tolerance
        freqs = np.array([1.0, -5.0])
        mock_phonopy.get_mesh_dict.return_value = {"frequencies": freqs}

        validator = PhononValidator(mock_validation_config, "pot.yace")
        with patch("mlip_autopipec.physics.validation.phonon.get_lammps_calculator"):
            result = validator.validate(mock_structure)

        assert result.overall_status == "FAIL"
