from unittest.mock import patch

import numpy as np
import pytest

from mlip_autopipec.components.validator.standard import StandardValidator
from mlip_autopipec.domain_models.config import StandardValidatorConfig
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure


@pytest.fixture
def potential(tmp_path):
    p = tmp_path / "pot.yace"
    p.touch()
    return Potential(path=p, species=["Si"])


@pytest.fixture
def validator_config():
    return StandardValidatorConfig(
        name="standard",
        phonon_supercell=[2, 2, 2],
        eos_strain_range=0.1,
        elastic_strain_magnitude=0.01
    )


def test_standard_validator_structure(validator_config):
    validator = StandardValidator(validator_config)
    assert validator.name == "standard"


@patch("mlip_autopipec.components.validator.standard.LammpsSinglePointCalculator")
@patch("mlip_autopipec.components.validator.standard.BFGS")
@patch("mlip_autopipec.components.validator.standard.UnitCellFilter")
@patch("mlip_autopipec.components.validator.standard.PhononCalc")
@patch("mlip_autopipec.components.validator.standard.ElasticCalc")
@patch("mlip_autopipec.components.validator.standard.EOSCalc")
def test_validate_all_pass(mock_eos, mock_elastic, mock_phonon, mock_ucf, mock_bfgs, mock_calc, validator_config, potential):
    # Setup Mocks
    mock_phonon_instance = mock_phonon.return_value
    mock_phonon_instance.calculate.return_value = (True, None)  # stable, no failed structure

    mock_elastic_instance = mock_elastic.return_value
    mock_elastic_instance.calculate.return_value = (True, 100.0, 50.0)  # stable, bulk, shear

    mock_eos_instance = mock_eos.return_value
    mock_eos_instance.calculate.return_value = 0.001  # low rmse

    # Mock relaxation
    mock_bfgs_instance = mock_bfgs.return_value
    # opt.run() does nothing, which is fine

    # Mock calculator methods
    mock_calc_instance = mock_calc.return_value
    mock_calc_instance.get_potential_energy.return_value = -10.0
    # Use side_effect to handle different number of atoms
    mock_calc_instance.get_forces.side_effect = lambda atoms=None: np.zeros((8, 3)) if atoms is None else np.zeros((len(atoms), 3))
    # Also handle the case where get_forces is called on the calculator without arguments (uses calc.atoms)
    # But Structure.from_ase calls atoms.get_forces(), which calls calc.get_forces(atoms).
    # Wait, calc.get_forces(atoms) signature.

    def get_forces_mock(atoms=None):
        if atoms is not None:
             return np.zeros((len(atoms), 3))
        # If atoms is None, we assume it's attached to calc.atoms but we can't easily access it in mock.
        # But Structure.from_ase calls atoms.get_forces() -> calc.get_forces(atoms).
        return np.zeros((8, 3)) # Fallback

    mock_calc_instance.get_forces.side_effect = get_forces_mock
    mock_calc_instance.get_stress.return_value = np.zeros(6)

    validator = StandardValidator(validator_config)
    metrics = validator.validate(potential)

    assert metrics.passed is True
    assert metrics.phonon_stable is True
    assert metrics.elastic_stable is True
    assert metrics.bulk_modulus == 100.0
    assert metrics.shear_modulus == 50.0
    assert metrics.eos_rmse == 0.001
    assert len(metrics.failed_structures) == 0


@patch("mlip_autopipec.components.validator.standard.LammpsSinglePointCalculator")
@patch("mlip_autopipec.components.validator.standard.BFGS")
@patch("mlip_autopipec.components.validator.standard.UnitCellFilter")
@patch("mlip_autopipec.components.validator.standard.PhononCalc")
@patch("mlip_autopipec.components.validator.standard.ElasticCalc")
@patch("mlip_autopipec.components.validator.standard.EOSCalc")
def test_validate_phonon_fail(mock_eos, mock_elastic, mock_phonon, mock_ucf, mock_bfgs, mock_calc, validator_config, potential):
    # Setup Mocks
    failed_struct = Structure(
        positions=np.array([[0, 0, 0]]),
        atomic_numbers=np.array([1]),
        cell=np.eye(3),
        pbc=np.array([True, True, True])
    )

    mock_phonon_instance = mock_phonon.return_value
    mock_phonon_instance.calculate.return_value = (False, failed_struct)

    mock_elastic_instance = mock_elastic.return_value
    mock_elastic_instance.calculate.return_value = (True, 100.0, 50.0)

    mock_eos_instance = mock_eos.return_value
    mock_eos_instance.calculate.return_value = 0.001

    # Mock calculator methods
    mock_calc_instance = mock_calc.return_value
    mock_calc_instance.get_potential_energy.return_value = -10.0
    mock_calc_instance.get_forces.side_effect = lambda atoms=None: np.zeros((8, 3)) if atoms is None else np.zeros((len(atoms), 3))
    mock_calc_instance.get_stress.return_value = np.zeros(6)

    validator = StandardValidator(validator_config)
    metrics = validator.validate(potential)

    assert metrics.passed is False
    assert metrics.phonon_stable is False
    assert len(metrics.failed_structures) == 1
    assert metrics.failed_structures[0] == failed_struct


@patch("mlip_autopipec.components.validator.standard.LammpsSinglePointCalculator")
@patch("mlip_autopipec.components.validator.standard.BFGS")
@patch("mlip_autopipec.components.validator.standard.UnitCellFilter")
@patch("mlip_autopipec.components.validator.standard.PhononCalc")
@patch("mlip_autopipec.components.validator.standard.ElasticCalc")
@patch("mlip_autopipec.components.validator.standard.EOSCalc")
def test_validate_elastic_fail(mock_eos, mock_elastic, mock_phonon, mock_ucf, mock_bfgs, mock_calc, validator_config, potential):
    # Setup Mocks
    mock_phonon_instance = mock_phonon.return_value
    mock_phonon_instance.calculate.return_value = (True, None)

    mock_elastic_instance = mock_elastic.return_value
    mock_elastic_instance.calculate.return_value = (False, None, None)

    mock_eos_instance = mock_eos.return_value
    mock_eos_instance.calculate.return_value = 0.001

    # Mock calculator methods
    mock_calc_instance = mock_calc.return_value
    mock_calc_instance.get_potential_energy.return_value = -10.0
    mock_calc_instance.get_forces.side_effect = lambda atoms=None: np.zeros((8, 3)) if atoms is None else np.zeros((len(atoms), 3))
    mock_calc_instance.get_stress.return_value = np.zeros(6)

    validator = StandardValidator(validator_config)
    metrics = validator.validate(potential)

    assert metrics.passed is False
    assert metrics.elastic_stable is False
