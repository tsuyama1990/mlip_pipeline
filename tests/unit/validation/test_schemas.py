import pytest
from pydantic import ValidationError

from mlip_autopipec.config.schemas.validation import (
    ElasticityConfig,
    EOSConfig,
    PhononConfig,
    ValidationConfig,
)


def test_validation_config_defaults():
    config = ValidationConfig()
    assert config.phonon.enabled is True
    assert config.elasticity.enabled is True
    assert config.eos.enabled is True


def test_phonon_config():
    config = PhononConfig(supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]], displacement=0.02)
    assert config.displacement == 0.02
    assert config.supercell_matrix == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_elasticity_config():
    config = ElasticityConfig(strain_max=0.01, num_points=7)
    assert config.strain_max == 0.01
    assert config.num_points == 7


def test_eos_config():
    config = EOSConfig(strain_max=0.2, num_points=9)
    assert config.strain_max == 0.2
    assert config.num_points == 9


def test_invalid_config():
    with pytest.raises(ValidationError):
        PhononConfig(extra_field="invalid")
