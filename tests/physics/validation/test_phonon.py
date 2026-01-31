from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from mlip_autopipec.domain_models.config import PotentialConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import ValidationConfig
from mlip_autopipec.physics.validation.phonon import PhononValidator


@pytest.fixture
def structure() -> Structure:
    atoms = Atoms(
        "Si",
        positions=[[0, 0, 0]],
        cell=[[0, 2.7, 2.7], [2.7, 0, 2.7], [2.7, 2.7, 0]],
        pbc=True,
    )
    return Structure.from_ase(atoms)


@pytest.fixture
def pot_config() -> PotentialConfig:
    return PotentialConfig(elements=["Si"], cutoff=5.0)


@patch("mlip_autopipec.physics.validation.phonon.Phonopy")
@patch("mlip_autopipec.physics.validation.phonon.get_validation_calculator")
def test_phonon_validate_success(
    mock_get_calc: MagicMock,
    MockPhonopy: MagicMock,
    structure: Structure,
    pot_config: PotentialConfig,
    tmp_path: Path,
) -> None:
    mock_phonopy_instance = MockPhonopy.return_value

    # Mock band structure
    frequencies = [
        [0.0, 0.0, 0.0, 5.0, 10.0, 15.0],
        [2.0, 2.0, 2.0, 6.0, 11.0, 16.0],
    ]
    mock_phonopy_instance.get_mesh_dict.return_value = {  # Updated to get_mesh_dict as implementation uses it
        "frequencies": frequencies,
        "qpoints": [],
        "distances": [],
    }

    config = ValidationConfig()
    validator = PhononValidator(config, pot_config, work_dir=tmp_path)

    metrics, plots = validator.validate(structure, potential_path=Path("pot.yace"))

    assert any(m.name == "Phonon Stability" and m.passed for m in metrics)
    assert "phonon_band_structure" in plots


@patch("mlip_autopipec.physics.validation.phonon.Phonopy")
@patch("mlip_autopipec.physics.validation.phonon.get_validation_calculator")
def test_phonon_validate_fail(
    mock_get_calc: MagicMock,
    MockPhonopy: MagicMock,
    structure: Structure,
    pot_config: PotentialConfig,
    tmp_path: Path,
) -> None:
    mock_phonopy_instance = MockPhonopy.return_value

    frequencies = [
        [0.0, 0.0, 0.0, -5.0, 10.0, 15.0],
    ]
    mock_phonopy_instance.get_mesh_dict.return_value = {
        "frequencies": frequencies,
        "qpoints": [],
        "distances": [],
    }

    config = ValidationConfig()
    validator = PhononValidator(config, pot_config, work_dir=tmp_path)

    metrics, plots = validator.validate(structure, potential_path=Path("pot.yace"))

    assert any(m.name == "Phonon Stability" and not m.passed for m in metrics)
