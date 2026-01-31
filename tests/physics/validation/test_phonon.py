import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np
from ase import Atoms

from mlip_autopipec.physics.validation.phonon import PhononValidator
from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig
from mlip_autopipec.domain_models.structure import Structure

def test_phonon_validator_pass():
    val_config = ValidationConfig(phonon_tolerance=-0.1)
    pot_config = PotentialConfig(elements=["Si"], cutoff=5.0)
    validator = PhononValidator(val_config, pot_config, potential_path=Path("dummy.yace"))

    # Mock internal methods to avoid heavy Phonopy dependency and calculation in unit test
    validator._setup_phonopy = MagicMock()
    validator._calculate_forces = MagicMock()
    validator._get_frequencies = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))
    validator._plot_band_structure = MagicMock(return_value=Path("phonon.png"))

    # Also need to mock produce_force_constants on the phonopy object returned by setup
    mock_phonopy = validator._setup_phonopy.return_value
    mock_phonopy.produce_force_constants = MagicMock()

    structure = Structure.from_ase(Atoms("Si", cell=[5,5,5], pbc=True))

    metric, plot_path = validator.validate(structure)

    assert metric.passed is True
    assert plot_path == Path("phonon.png")

def test_phonon_validator_fail():
    val_config = ValidationConfig(phonon_tolerance=-0.1)
    pot_config = PotentialConfig(elements=["Si"], cutoff=5.0)
    validator = PhononValidator(val_config, pot_config, potential_path=Path("dummy.yace"))

    # Return imaginary frequencies (represented as negative numbers)
    validator._setup_phonopy = MagicMock()
    validator._calculate_forces = MagicMock()
    validator._get_frequencies = MagicMock(return_value=np.array([1.0, -1.0, 3.0]))
    validator._plot_band_structure = MagicMock(return_value=Path("phonon.png"))

    mock_phonopy = validator._setup_phonopy.return_value
    mock_phonopy.produce_force_constants = MagicMock()

    structure = Structure.from_ase(Atoms("Si", cell=[5,5,5], pbc=True))

    metric, _ = validator.validate(structure)

    assert metric.passed is False
