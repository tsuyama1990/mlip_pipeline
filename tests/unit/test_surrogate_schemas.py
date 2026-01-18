import pytest
from pydantic import ValidationError

from mlip_autopipec.config.schemas.surrogate import SelectionResult, SurrogateConfig


def test_surrogate_config_defaults():
    config = SurrogateConfig()
    assert config.model_path == "medium"
    assert config.device == "cuda"
    assert config.fps_n_samples == 100
    assert config.force_threshold == 50.0
    assert config.descriptor_type == "soap"


def test_surrogate_config_validation():
    # Test invalid device
    with pytest.raises(ValidationError):
        SurrogateConfig(device="tpu")

    # Test invalid fps_n_samples
    with pytest.raises(ValidationError):
        SurrogateConfig(fps_n_samples=0)

    # Test invalid force_threshold
    with pytest.raises(ValidationError):
        SurrogateConfig(force_threshold=-1.0)

    # Test extra fields
    with pytest.raises(ValidationError):
        SurrogateConfig(extra_field="invalid")


def test_selection_result_validation():
    # Test valid result
    res = SelectionResult(selected_indices=[1, 2, 3], scores=[0.1, 0.2, 0.3])
    assert res.selected_indices == [1, 2, 3]
    assert res.scores == [0.1, 0.2, 0.3]

    # Test invalid types
    with pytest.raises(ValidationError):
        SelectionResult(selected_indices=["a"], scores=[0.1])
