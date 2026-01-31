from unittest.mock import patch

import pytest
import numpy as np
from mlip_autopipec.domain_models.config import ValidationConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.validation.elasticity import ElasticityValidator


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


def test_elasticity_validator_pass(mock_validation_config, mock_structure):
    with patch.object(ElasticityValidator, "_calculate_stiffness") as mock_calc:
        # Cubic stability: C11 - C12 > 0, C11 + 2C12 > 0, C44 > 0
        # Stable
        C = np.zeros((6, 6))
        C[0, 0] = 160  # C11
        C[1, 1] = 160
        C[2, 2] = 160
        C[0, 1] = 60  # C12
        C[1, 0] = 60
        C[3, 3] = 80  # C44
        mock_calc.return_value = C

        validator = ElasticityValidator(mock_validation_config, "pot.yace")
        # We need to mock get_lammps_calculator to avoid errors
        with patch(
            "mlip_autopipec.physics.validation.elasticity.get_lammps_calculator"
        ):
            result = validator.validate(mock_structure)

        assert result.overall_status == "PASS"


def test_elasticity_validator_fail(mock_validation_config, mock_structure):
    with patch.object(ElasticityValidator, "_calculate_stiffness") as mock_calc:
        # Unstable: C44 < 0
        C = np.zeros((6, 6))
        C[3, 3] = -10
        mock_calc.return_value = C

        validator = ElasticityValidator(mock_validation_config, "pot.yace")
        with patch(
            "mlip_autopipec.physics.validation.elasticity.get_lammps_calculator"
        ):
            result = validator.validate(mock_structure)

        assert result.overall_status == "FAIL"
