import pytest
from pydantic import ValidationError

from mlip_autopipec.config.schemas.validation import (
    ElasticConfig,
    EOSConfig,
    PhononConfig,
    ValidationConfig,
)


def test_phonon_config_defaults():
    config = PhononConfig()
    assert config.enabled is True
    assert config.supercell_matrix == [2, 2, 2]
    assert config.displacement == 0.01
    assert config.symprec == 1e-5


def test_phonon_config_validation():
    # Test valid
    config = PhononConfig(supercell_matrix=[3, 3, 3], displacement=0.02)
    assert config.supercell_matrix == [3, 3, 3]

    # Test invalid supercell length
    with pytest.raises(ValidationError):
        PhononConfig(supercell_matrix=[2, 2])

    # Test invalid displacement
    with pytest.raises(ValidationError):
        PhononConfig(displacement=-0.1)


def test_elastic_config_defaults():
    config = ElasticConfig()
    assert config.enabled is True
    assert config.num_points == 5
    assert config.max_distortion == 0.05


def test_elastic_config_validation():
    # Test min points
    with pytest.raises(ValidationError):
        ElasticConfig(num_points=2)

    # Test max distortion gt 0
    with pytest.raises(ValidationError):
        ElasticConfig(max_distortion=0.0)


def test_eos_config_defaults():
    config = EOSConfig()
    assert config.enabled is True
    assert config.num_points == 10
    assert config.strain_max == 0.1


def test_eos_config_validation():
    with pytest.raises(ValidationError):
        EOSConfig(num_points=4)


def test_validation_config_defaults():
    config = ValidationConfig()
    assert config.phonon.enabled is True
    assert config.elastic.enabled is True
    assert config.eos.enabled is True
    assert config.fail_on_instability is False
    assert config.reference_data is None


def test_validation_config_extra_forbid():
    with pytest.raises(ValidationError):
        ValidationConfig(extra_field="fail")


def test_validation_config_reference_data():
    data = {"bulk_modulus": 100.0}
    config = ValidationConfig(reference_data=data)
    assert config.reference_data == data
