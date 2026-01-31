from unittest.mock import MagicMock, patch
import pytest
import numpy as np
from mlip_autopipec.physics.validation.phonon import PhononValidator
from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig
from mlip_autopipec.domain_models.structure import Structure

@pytest.fixture
def mock_calc():
    with patch("mlip_autopipec.physics.validation.phonon.get_lammps_calculator") as mock:
        calc = MagicMock()
        # Phonon needs forces
        calc.get_forces.return_value = np.zeros((2, 3))
        mock.return_value = calc
        yield calc

def test_phonon_validator_pass(tmp_path):
    # Create dummy potential file
    (tmp_path / "pot.yace").touch()

    val_config = ValidationConfig(phonon_tolerance=-0.1)
    pot_config = PotentialConfig(elements=["Si"], cutoff=5.0, pair_style="hybrid/overlay")
    validator = PhononValidator(val_config, pot_config, tmp_path / "pot.yace")

    structure = Structure(
        symbols=["Si"]*2,
        positions=np.zeros((2,3)),
        cell=np.eye(3),
        pbc=(True,True,True)
    )

    # Mock phonopy
    with patch("mlip_autopipec.physics.validation.phonon.Phonopy") as mock_phonopy_cls:
        mock_phonopy = MagicMock()
        mock_phonopy_cls.return_value = mock_phonopy

        # Mock mesh frequencies
        mock_phonopy.get_mesh_dict.return_value = {
            'frequencies': np.array([0.1, 0.2, 0.1, 0.2])
        }

        # We also need to mock get_lammps_calculator to accept the tmp path
        with patch("mlip_autopipec.physics.validation.phonon.get_lammps_calculator") as mock_calc_fac:
             calc = MagicMock()
             calc.get_forces.return_value = np.zeros((2, 3))
             mock_calc_fac.return_value = calc

             metric, plot = validator.validate(structure)
             assert metric.passed is True

def test_phonon_validator_fail(tmp_path):
    (tmp_path / "pot.yace").touch()

    val_config = ValidationConfig(phonon_tolerance=-0.1)
    pot_config = PotentialConfig(elements=["Si"], cutoff=5.0, pair_style="hybrid/overlay")
    validator = PhononValidator(val_config, pot_config, tmp_path / "pot.yace")

    structure = Structure(
        symbols=["Si"]*2,
        positions=np.zeros((2,3)),
        cell=np.eye(3),
        pbc=(True,True,True)
    )

    with patch("mlip_autopipec.physics.validation.phonon.Phonopy") as mock_phonopy_cls:
        mock_phonopy = MagicMock()
        mock_phonopy_cls.return_value = mock_phonopy

        # Mock mesh frequencies with unstable mode
        mock_phonopy.get_mesh_dict.return_value = {
            'frequencies': np.array([-0.5, 0.2, -0.5, 0.2])
        }

        with patch("mlip_autopipec.physics.validation.phonon.get_lammps_calculator") as mock_calc_fac:
             calc = MagicMock()
             calc.get_forces.return_value = np.zeros((2, 3))
             mock_calc_fac.return_value = calc

             metric, plot = validator.validate(structure)
             assert metric.passed is False
