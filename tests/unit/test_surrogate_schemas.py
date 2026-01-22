import pytest
from pydantic import ValidationError

from mlip_autopipec.config.schemas.surrogate import SurrogateConfig


def test_surrogate_config_valid():
    config = SurrogateConfig(
        model_type="mace_mp",
        model_path="medium",
        device="cuda",
        force_threshold=100.0,
        n_samples=50
    )
    assert config.model_type == "mace_mp"
    assert config.model_path == "medium"
    assert config.device == "cuda"
    assert config.force_threshold == 100.0
    assert config.n_samples == 50

def test_surrogate_config_defaults():
    config = SurrogateConfig()
    assert config.model_type == "mace_mp"
    assert config.model_path == "medium"
    assert config.device == "cpu"
    assert config.force_threshold == 50.0
    assert config.n_samples == 100

def test_surrogate_config_extra_forbid():
    with pytest.raises(ValidationError):
        SurrogateConfig(extra_field="invalid")

def test_surrogate_config_types():
    with pytest.raises(ValidationError):
        SurrogateConfig(force_threshold="invalid")
