import pytest
from pydantic import ValidationError

from mlip_autopipec.config.schemas.validation import (
    ElasticConfig,
    EOSConfig,
    PhononConfig,
    ValidationConfig,
)


def test_phonon_config_valid():
    config = PhononConfig(supercell_matrix=[3, 3, 3], displacement=0.02)
    assert config.supercell_matrix == [3, 3, 3]
    assert config.displacement == 0.02


def test_phonon_config_invalid_matrix():
    with pytest.raises(ValidationError):
        PhononConfig(supercell_matrix=[2, 2])  # Too short


def test_elastic_config_valid():
    config = ElasticConfig(num_points=5, max_distortion=0.01)
    assert config.num_points == 5


def test_elastic_config_invalid_points():
    with pytest.raises(ValidationError):
        ElasticConfig(num_points=2)  # Too few


def test_eos_config_valid():
    config = EOSConfig(num_points=10, strain_max=0.05)
    assert config.num_points == 10


def test_validation_config_defaults():
    config = ValidationConfig()
    assert config.phonon.supercell_matrix == [2, 2, 2]
    assert config.elastic.num_points == 5
    assert not config.fail_on_instability
