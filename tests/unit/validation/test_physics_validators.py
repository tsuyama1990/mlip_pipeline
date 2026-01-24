from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms
from ase.units import GPa

from mlip_autopipec.config.schemas.validation import (
    ElasticConfig,
    EOSConfig,
    PhononConfig,
    ValidationConfig,
)
from mlip_autopipec.data_models.validation import ValidationResult
from mlip_autopipec.validation.elasticity import ElasticityValidator
from mlip_autopipec.validation.eos import EOSValidator
from mlip_autopipec.validation.phonon import PhononValidator
from mlip_autopipec.validation.runner import ValidationRunner


@pytest.fixture
def atoms():
    a = Atoms('Al', positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True)
    a.calc = MagicMock()
    a.calc.get_potential_energy.return_value = -10.0
    a.calc.get_forces.return_value = np.zeros((1, 3))
    a.calc.get_stress.return_value = np.zeros(6)
    return a

@patch("mlip_autopipec.validation.phonon.Phonopy")
def test_phonon_validator_stable(mock_phonopy_cls, atoms):
    mock_phonopy = mock_phonopy_cls.return_value
    # Mock mesh dict (used in code)
    mock_phonopy.get_mesh_dict.return_value = {
        "frequencies": [[1.0, 2.0], [1.5, 2.5]],
        "qpoints": [[0,0,0], [0.5,0,0]]
    }

    validator = PhononValidator(PhononConfig(supercell_matrix=[2,2,2]))
    result = validator.validate(atoms)

    assert result.passed
    assert result.module == "phonon"
    assert not result.error

@patch("mlip_autopipec.validation.phonon.Phonopy")
def test_phonon_validator_unstable(mock_phonopy_cls, atoms):
    mock_phonopy = mock_phonopy_cls.return_value
    # Imaginary mode (negative frequency)
    mock_phonopy.get_mesh_dict.return_value = {
        "frequencies": [[-1.0, 2.0], [1.5, 2.5]],
        "qpoints": [[0,0,0], [0.5,0,0]]
    }

    validator = PhononValidator(PhononConfig())
    result = validator.validate(atoms)

    assert not result.passed
    # We check for metric name "min_frequency" and passed=False
    assert not result.metrics[0].passed
    assert result.metrics[0].value == -1.0

@patch("mlip_autopipec.validation.eos.EquationOfState")
def test_eos_validator(mock_eos_cls, atoms):
    mock_eos = mock_eos_cls.return_value
    # fit() returns v0, e0, B
    # B should be in eV/A^3 if we want B_GPa = B/GPa to equal 70
    mock_eos.fit.return_value = (27.0, -10.0, 70.0 * GPa)
    mock_eos.v0 = 27.0
    mock_eos.e0 = -10.0
    mock_eos.B = 70.0 * GPa

    validator = EOSValidator(EOSConfig())

    result = validator.validate(atoms)

    assert result.passed
    assert result.module == "eos"
    # Check close enough floating point
    bulk_mod = next(m for m in result.metrics if m.name == "bulk_modulus")
    assert pytest.approx(bulk_mod.value) == 70.0

def test_elasticity_validator_stub(atoms):
    validator = ElasticityValidator(ElasticConfig())
    with patch.object(validator, '_calculate_stiffness_matrix', return_value=np.eye(6)*100):
        result = validator.validate(atoms)
        assert result.passed
        assert result.module == "elastic"

def test_elasticity_calculation(atoms):
    config = ElasticConfig(max_distortion=0.01)
    validator = ElasticityValidator(config)

    # We want to mock get_stress to simulate a material with C_11 = 100 GPa
    # Stress = C * Strain
    # When we apply strain e_xx = +/- 0.01
    # Stress s_xx = 100 * (+/- 0.01) = +/- 1.0 GPa
    # ASE stress is in eV/A^3. 1 GPa = 0.00624 eV/A^3.
    # So we need to return +/- 1.0 * 0.00624

    target_C11 = 100.0
    stress_val = target_C11 * config.max_distortion * 0.0062415

    # We create a side effect that returns positive then negative stress for the first component
    # and zero for others.
    # The loop runs 6 times (xx, yy, zz...). Inside each, 2 calls (+, -).
    # Total 12 calls.
    # We only care about the first iteration (xx) to verify C[0,0].

    responses = []
    # For i=0 (xx):
    # Inner loop is [-delta, +delta]
    responses.append(np.array([-stress_val, 0, 0, 0, 0, 0])) # -delta
    responses.append(np.array([stress_val, 0, 0, 0, 0, 0])) # +delta
    # For others i=1..5: return zeros
    for _ in range(10):
        responses.append(np.zeros(6))

    atoms.calc.get_stress.side_effect = responses

    C = validator._calculate_stiffness_matrix(atoms)

    # Check C[0,0] (xx, xx)
    # The calculated value should be close to target_C11
    # Our validator converts result back to GPa.
    # Result = (sigma+ - sigma-) / 2delta  [in GPa]
    # sigma in GPa = (stress_val / 0.0062415) = 1.0
    # Diff = 2.0. / 0.02 = 100.

    # Note: validator divides by GPa inside the loop.
    # stress_gpa = stress / GPa.
    # 0.00624 eV/A^3 / GPa = 1.0 (approx).

    assert pytest.approx(C[0,0], abs=1.0) == target_C11

def test_validation_runner(atoms):
    config = ValidationConfig()
    runner = ValidationRunner(config)

    # Mock validators
    with patch("mlip_autopipec.validation.runner.PhononValidator") as mock_ph, \
         patch("mlip_autopipec.validation.runner.ElasticityValidator") as mock_el, \
         patch("mlip_autopipec.validation.runner.EOSValidator") as mock_eos:

        mock_ph.return_value.validate.return_value = ValidationResult(module="phonon", passed=True)
        mock_el.return_value.validate.return_value = ValidationResult(module="elastic", passed=True)
        mock_eos.return_value.validate.return_value = ValidationResult(module="eos", passed=True)

        results = runner.run(atoms, modules=["phonon", "eos"])

        assert len(results) == 2
        assert results[0].module == "phonon"
        assert results[1].module == "eos"
        mock_ph.assert_called_once()
        mock_eos.assert_called_once()
        mock_el.assert_not_called()
