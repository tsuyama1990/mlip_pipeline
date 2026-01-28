import pytest
from pydantic import ValidationError

from mlip_autopipec.config.schemas.inference import EONConfig, InferenceConfig


def test_eon_config_defaults():
    config = EONConfig()
    assert config.temperature == 300.0
    assert config.job == "process_search"
    assert config.pot_name == "pace_driver"
    assert config.parameters == {}


def test_eon_config_validation():
    # Test valid
    config = EONConfig(temperature=500.0, job="saddle_search")
    assert config.temperature == 500.0
    assert config.job == "saddle_search"

    # Test invalid temperature
    with pytest.raises(ValidationError):
        EONConfig(temperature=-10.0)

    # Test invalid job
    with pytest.raises(ValidationError):
        EONConfig(job="invalid_job")


def test_inference_config_integration():
    # Default should have active_engine="lammps" and eon=None
    config = InferenceConfig()
    assert config.active_engine == "lammps"
    assert config.eon is None

    # Test setting eon
    eon_conf = EONConfig(temperature=600.0)
    config = InferenceConfig(active_engine="eon", eon=eon_conf)
    assert config.active_engine == "eon"
    assert config.eon.temperature == 600.0
