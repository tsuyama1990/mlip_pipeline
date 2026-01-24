from unittest.mock import MagicMock

from ase import Atoms

from mlip_autopipec.config.schemas.validation import EOSConfig
from mlip_autopipec.validation.eos import EOSValidator


def test_eos_validator_success():
    config = EOSConfig(enabled=True, strain_max=0.05, num_points=5)
    validator = EOSValidator(config)
    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
    calculator = MagicMock()

    # Mock energies: parabolic well
    # Volumes will be roughly 0.95, 0.975, 1.0, 1.025, 1.05 * V0
    # Energies: high, low, min, low, high
    calculator.get_potential_energy.side_effect = [1.0, 0.2, 0.0, 0.2, 1.0]

    result = validator.validate(atoms, calculator)
    assert result is True
    assert calculator.get_potential_energy.call_count == 5


def test_eos_validator_failure_unstable():
    # If B0 is negative or fitting fails
    config = EOSConfig(enabled=True, strain_max=0.05, num_points=5)
    validator = EOSValidator(config)
    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
    calculator = MagicMock()

    # Inverted parabola (unstable volume)
    calculator.get_potential_energy.side_effect = [-1.0, -0.2, 0.0, -0.2, -1.0]

    result = validator.validate(atoms, calculator)
    assert result is False
