from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import ValidatorConfig
from mlip_autopipec.domain_models.enums import ValidatorType


def test_validator_config_defaults() -> None:
    """Test default values for ValidatorConfig."""
    config = ValidatorConfig()
    assert config.type == ValidatorType.MOCK
    assert config.elastic_tolerance == 0.15
    assert config.phonon_stability is True
    assert config.phonon_supercell == [2, 2, 2]
    assert config.strain_magnitude == 0.01
    assert config.structure_path is None

def test_validator_config_custom_values() -> None:
    """Test custom values for ValidatorConfig."""
    structure_path = Path("/valid/path/structure.xyz")
    config = ValidatorConfig(
        type=ValidatorType.PHYSICS,
        elastic_tolerance=0.05,
        phonon_stability=False,
        phonon_supercell=[3, 3, 3],
        strain_magnitude=0.005,
        structure_path=structure_path
    )
    assert config.type == ValidatorType.PHYSICS
    assert config.elastic_tolerance == 0.05
    assert config.phonon_stability is False
    assert config.phonon_supercell == [3, 3, 3]
    assert config.strain_magnitude == 0.005
    assert config.structure_path == structure_path

def test_validator_config_validation_error() -> None:
    """Test validation errors."""
    with pytest.raises(ValidationError):
        ValidatorConfig(elastic_tolerance=-0.1)

    with pytest.raises(ValidationError):
        ValidatorConfig(strain_magnitude=0.0)

    # Supercell validation? Pydantic just checks list[int].
    # But usually supercell should be positive.
    # The config doesn't have strict validator for contents of list, just type.
    config = ValidatorConfig(phonon_supercell=[1, 1, 1])
    assert config.phonon_supercell == [1, 1, 1]
