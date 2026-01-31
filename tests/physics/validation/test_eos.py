import pytest
from ase import Atoms
from unittest.mock import MagicMock
from pathlib import Path

from mlip_autopipec.physics.validation.eos import EOSValidator
from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig
from mlip_autopipec.domain_models.structure import Structure

@pytest.fixture
def mock_calc():
    """Mock calculator that behaves like a harmonic potential near equilibrium."""
    calc = MagicMock()
    # To simulate Birch-Murnaghan, we need energy as a function of volume
    # Let's just return values that fit a parabola
    # We will mock the behavior inside the validator or assume the validator calls calc.get_potential_energy(atoms)

    def side_effect(atoms=None):
        if atoms is None:
             # If called without atoms (attached), use calc.atoms
             atoms = calc.atoms
        vol = atoms.get_volume()
        v0 = 27.0 # 3^3
        k = 0.1
        return k * (vol - v0)**2 - 100.0 # Shifted parabola

    calc.get_potential_energy.side_effect = side_effect
    return calc

def test_eos_validator_success(mock_calc):
    # Setup
    val_config = ValidationConfig()
    pot_config = PotentialConfig(elements=["Si"], cutoff=5.0)

    validator = EOSValidator(val_config, pot_config, potential_path=Path("dummy.yace"))
    # Mock the calculator factory
    validator._get_calculator = MagicMock(return_value=mock_calc)

    # Create dummy structure
    atoms = Atoms("Si1", positions=[[0,0,0]], cell=[[3,0,0],[0,3,0],[0,0,3]], pbc=True)
    structure = Structure.from_ase(atoms)

    # Execute
    metric = validator.validate(structure)

    # Verify
    assert metric.name == "Bulk Modulus (EOS)"
    assert metric.passed is True
    assert metric.value > 0  # Bulk modulus should be positive
    assert metric.error is None
