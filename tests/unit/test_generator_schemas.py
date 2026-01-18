import pytest
from pydantic import ValidationError

from mlip_autopipec.config.schemas.generator import (
    DefectConfig,
    DistortionConfig,
    GeneratorConfig,
    NMSConfig,
    SQSConfig,
)


def test_generator_config_defaults():
    # Now we must provide arguments because defaults were removed from schema
    config = GeneratorConfig(
        sqs=SQSConfig(), distortion=DistortionConfig(), nms=NMSConfig(), defects=DefectConfig()
    )

    # SQS
    assert config.sqs.enabled is True
    assert config.sqs.supercell_matrix == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

    # Distortion
    assert config.distortion.enabled is True
    assert config.distortion.rattling_amplitude == 0.05
    assert config.distortion.strain_range == (-0.05, 0.05)
    assert config.distortion.n_strain_steps == 5
    assert config.distortion.n_rattle_steps == 3

    # NMS
    assert config.nms.enabled is True
    assert config.nms.temperatures == [300, 600, 1000]

    # Defects
    assert config.defects.enabled is False  # Default is false
    assert config.defects.vacancies is True


def test_generator_config_validation():
    # rattling_amplitude must be > 0
    with pytest.raises(ValidationError):
        GeneratorConfig(
            sqs=SQSConfig(),
            distortion=DistortionConfig(rattling_amplitude=0.0),
            nms=NMSConfig(),
            defects=DefectConfig(),
        )

    # n_strain_steps must be >= 1
    with pytest.raises(ValidationError):
        GeneratorConfig(
            sqs=SQSConfig(),
            distortion=DistortionConfig(n_strain_steps=0),
            nms=NMSConfig(),
            defects=DefectConfig(),
        )

    # Extra fields not allowed
    with pytest.raises(ValidationError):
        GeneratorConfig(
            sqs=SQSConfig(),
            distortion=DistortionConfig(),
            nms=NMSConfig(),
            defects=DefectConfig(),
            extra_field="invalid",
        )
