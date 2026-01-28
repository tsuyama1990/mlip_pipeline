import pytest
from pydantic import ValidationError

from mlip_autopipec.config.schemas.generator import (
    DefectConfig,
    DistortionConfig,
    GeneratorConfig,
    SQSConfig,
)


def test_sqs_config_defaults() -> None:
    config = SQSConfig()
    assert config.enabled is True
    assert config.supercell_size == [2, 2, 2]


def test_distortion_config_defaults() -> None:
    config = DistortionConfig()
    assert config.enabled is True
    assert config.strain_range == (-0.05, 0.05)
    assert config.rattle_stdev == 0.01


def test_defect_config_defaults() -> None:
    config = DefectConfig()
    assert config.enabled is False
    assert config.vacancies is False
    assert config.interstitials is False
    assert config.interstitial_elements == []


def test_generator_config_defaults() -> None:
    config = GeneratorConfig()
    assert isinstance(config.sqs, SQSConfig)
    assert isinstance(config.distortion, DistortionConfig)
    assert isinstance(config.defects, DefectConfig)
    assert config.number_of_structures == 10
    assert config.seed is None


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        GeneratorConfig(extra_field="invalid")  # type: ignore[call-arg]


def test_types_validation() -> None:
    with pytest.raises(ValidationError):
        GeneratorConfig(number_of_structures="ten")  # type: ignore[arg-type] # Should be int
