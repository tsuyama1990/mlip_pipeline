from mlip_autopipec.domain_models.config import ValidatorConfig
from mlip_autopipec.domain_models.enums import ValidatorType


def test_validator_config_defaults() -> None:
    config = ValidatorConfig()
    assert config.type == ValidatorType.MOCK
    assert config.elastic_tolerance == 0.15
    assert config.phonon_stability is True
    assert config.phonon_supercell == [2, 2, 2]
    assert config.strain_magnitude == 0.01

def test_validator_config_custom_values() -> None:
    config = ValidatorConfig(
        type=ValidatorType.PHYSICS,
        elastic_tolerance=0.1,
        phonon_stability=False,
        phonon_supercell=[3, 3, 3],
        strain_magnitude=0.02
    )
    assert config.type == ValidatorType.PHYSICS
    assert config.elastic_tolerance == 0.1
    assert config.phonon_stability is False
    assert config.phonon_supercell == [3, 3, 3]
    assert config.strain_magnitude == 0.02

def test_validator_config_validation_error() -> None:
    import pytest
    from pydantic import ValidationError

    # Test invalid strain magnitude
    with pytest.raises(ValidationError):
        ValidatorConfig(strain_magnitude=-0.01)

    # Test invalid phonon supercell length
    with pytest.raises(ValidationError):
        ValidatorConfig(phonon_supercell=[2, 2])
