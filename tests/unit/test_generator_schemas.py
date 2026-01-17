import pytest
from pydantic import ValidationError

from mlip_autopipec.config.schemas.generator import GeneratorConfig


def test_generator_config_defaults():
    config = GeneratorConfig()
    assert config.supercell_matrix == [[2,0,0], [0,2,0], [0,0,2]]
    assert config.rattling_amplitude == 0.05
    assert config.strain_range == (-0.05, 0.05)
    assert config.n_strain_steps == 5
    assert config.n_rattle_steps == 3
    assert config.temperatures == [300, 600, 1000]

def test_generator_config_validation():
    # rattling_amplitude must be > 0
    with pytest.raises(ValidationError):
        GeneratorConfig(rattling_amplitude=0.0)

    # n_strain_steps must be >= 1
    with pytest.raises(ValidationError):
        GeneratorConfig(n_strain_steps=0)

    # Extra fields not allowed
    with pytest.raises(ValidationError):
        GeneratorConfig(extra_field="invalid")
