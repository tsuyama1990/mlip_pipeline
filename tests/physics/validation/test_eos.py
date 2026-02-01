import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from mlip_autopipec.physics.validation.eos import EOSValidator
from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig, ACEConfig
from mlip_autopipec.domain_models.structure import Structure
import numpy as np

@pytest.fixture
def mock_calc():
    with patch("mlip_autopipec.physics.validation.eos.get_lammps_calculator") as mock:
        calc = MagicMock()
        calc.get_potential_energy.side_effect = lambda atoms=None: (atoms.get_volume()-64.0)**2 if atoms else 0.0
        mock.return_value = calc
        yield calc

def test_eos_validator_success(mock_calc):
    # Setup
    val_config = ValidationConfig()
    pot_config = PotentialConfig(
        elements=["Si"],
        cutoff=5.0,
        pair_style="hybrid/overlay",
        ace_params=ACEConfig(
            npot="FinnisSinclair",
            fs_parameters=[1, 1, 1, 0.5],
            ndensity=2
        )
    )
    validator = EOSValidator(val_config, pot_config, Path("pot.yace"))

    structure = Structure(
        symbols=["Si"]*8,
        positions=np.zeros((8,3)),
        cell=np.eye(3)*4.0,
        pbc=(True,True,True)
    )

    metric = validator.validate(structure)
    assert metric.passed is True
    assert metric.value > 0 # Bulk Modulus should be positive
