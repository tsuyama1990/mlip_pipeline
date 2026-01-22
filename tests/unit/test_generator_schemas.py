import pytest
from pydantic import ValidationError

from mlip_autopipec.config.schemas.generator import (
    DefectConfig,
    DistortionConfig,
    GeneratorConfig,
    SQSConfig,
)


def test_sqs_config_defaults():
    config = SQSConfig()
    assert config.enabled is True
    assert config.supercell_size == [2, 2, 2]


def test_sqs_config_valid():
    config = SQSConfig(enabled=False, supercell_size=[3, 3, 3])
    assert config.enabled is False
    assert config.supercell_size == [3, 3, 3]


def test_sqs_config_extra_forbid():
    with pytest.raises(ValidationError):
        SQSConfig(extra_field="fail")


def test_distortion_config_defaults():
    config = DistortionConfig()
    assert config.enabled is True
    assert config.strain_range == (-0.05, 0.05)
    assert config.rattle_stdev == 0.01


def test_distortion_config_valid():
    config = DistortionConfig(enabled=False, rattle_stdev=0.05)
    assert config.rattle_stdev == 0.05


def test_defect_config_defaults():
    config = DefectConfig()
    assert config.enabled is False
    assert config.vacancies is False
    assert config.interstitials is False
    assert config.interstitial_elements == []


def test_generator_config_nested():
    config = GeneratorConfig(
        sqs=SQSConfig(enabled=False),
        distortion=DistortionConfig(strain_range=(-0.1, 0.1)),
        defects=DefectConfig(enabled=True, vacancies=True),
        number_of_structures=50,
    )
    assert config.sqs.enabled is False
    assert config.distortion.strain_range == (-0.1, 0.1)
    assert config.defects.vacancies is True
    assert config.number_of_structures == 50
