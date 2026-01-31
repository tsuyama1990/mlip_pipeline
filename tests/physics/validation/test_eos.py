from unittest.mock import MagicMock, patch

import pytest
import numpy as np
from mlip_autopipec.domain_models.config import ValidationConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.validation.eos import EOSValidator


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


def test_eos_validator_pass(mock_validation_config, mock_structure):
    """Test EOS validator with a perfect EOS curve."""
    with (
        patch("mlip_autopipec.physics.validation.eos.EquationOfState") as mock_eos_cls,
        patch(
            "mlip_autopipec.physics.validation.eos.get_lammps_calculator"
        ) as mock_calc_getter,
    ):
        # Mock Calculator
        mock_calc = MagicMock()
        mock_calc.get_potential_energy.return_value = -10.0
        mock_calc_getter.return_value = mock_calc

        # Mock EOS fitting result
        mock_eos_instance = MagicMock()
        # v0, e0, B, dB/dP (usually returned by fit)
        # We need to mock what EOS.fit() returns, OR usually we access properties after fit?
        # ase.eos.EquationOfState.fit() returns [e0, B, Bprime, v0]
        mock_eos_instance.fit.return_value = (-10.0, 1.0, 4.0, 1.0)
        # Actually ASE returns e0, B, Bp, v0.
        # Let's assume Bulk Modulus B is positive (1.0 eV/A^3 = 160 GPa approx)

        mock_eos_cls.return_value = mock_eos_instance

        validator = EOSValidator(mock_validation_config, potential_path="pot.yace")
        result = validator.validate(mock_structure)

        assert result.overall_status == "PASS"
        assert result.metrics[0].name == "Bulk Modulus"
        assert result.metrics[0].value > 0


def test_eos_validator_fail(mock_validation_config, mock_structure):
    """Test EOS validator with negative Bulk Modulus."""
    with (
        patch("mlip_autopipec.physics.validation.eos.EquationOfState") as mock_eos_cls,
        patch("mlip_autopipec.physics.validation.eos.get_lammps_calculator"),
    ):
        mock_eos_instance = MagicMock()
        # Negative Bulk Modulus
        mock_eos_instance.fit.return_value = (-10.0, -1.0, 4.0, 1.0)
        mock_eos_cls.return_value = mock_eos_instance

        validator = EOSValidator(mock_validation_config, potential_path="pot.yace")
        result = validator.validate(mock_structure)

        assert result.overall_status == "FAIL"
