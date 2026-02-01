import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from mlip_autopipec.physics.validation.elasticity import ElasticityValidator
from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig
from mlip_autopipec.domain_models.structure import Structure
import numpy as np

@pytest.fixture
def mock_calc():
    with patch("mlip_autopipec.physics.validation.elasticity.get_lammps_calculator") as mock:
        calc = MagicMock()
        calc.get_stress.return_value = np.zeros(6)
        calc.get_potential_energy.return_value = 0.0
        mock.return_value = calc
        yield calc

def test_elasticity_validator_success(mock_calc):
    val_config = ValidationConfig()
    pot_config = PotentialConfig(
        elements=["Al"],
        cutoff=5.0,
        pair_style="hybrid/overlay",
        npot="FinnisSinclair",
        fs_parameters=[1, 1, 1, 0.5],
        ndensity=2
    )
    validator = ElasticityValidator(val_config, pot_config, Path("pot.yace"))

    structure = Structure(
        symbols=["Al"]*4,
        positions=np.zeros((4,3)),
        cell=np.eye(3)*4.0,
        pbc=(True,True,True)
    )

    # Mock _calculate_stiffness to return a stable matrix (identity)
    with patch.object(validator, "_calculate_stiffness", return_value=np.eye(6)):
        metric = validator.validate(structure)
        assert metric.passed is True

def test_elasticity_validator_failure(mock_calc):
    val_config = ValidationConfig()
    pot_config = PotentialConfig(
        elements=["Al"],
        cutoff=5.0,
        pair_style="hybrid/overlay",
        npot="FinnisSinclair",
        fs_parameters=[1, 1, 1, 0.5],
        ndensity=2
    )
    validator = ElasticityValidator(val_config, pot_config, Path("pot.yace"))

    structure = Structure(
        symbols=["Al"]*4,
        positions=np.zeros((4,3)),
        cell=np.eye(3)*4.0,
        pbc=(True,True,True)
    )

    # Mock _calculate_stiffness to return an unstable matrix (negative identity)
    with patch.object(validator, "_calculate_stiffness", return_value=-np.eye(6)):
        metric = validator.validate(structure)
        assert metric.passed is False
