import pytest
from pydantic import ValidationError

from mlip_autopipec.config.schemas.inference import InferenceConfig


def test_inference_config_defaults():
    config = InferenceConfig()
    assert config.ensemble == "nvt"
    assert config.use_zbl_baseline is True
    assert config.uncertainty_threshold == 5.0
    assert config.sampling_interval == 10


def test_inference_config_valid():
    config = InferenceConfig(
        ensemble="npt", steps=5000, temperature=1000.0, pressure=10.0, use_zbl_baseline=False
    )
    assert config.ensemble == "npt"
    assert config.steps == 5000
    assert not config.use_zbl_baseline


def test_inference_config_invalid_ensemble():
    with pytest.raises(ValidationError):
        InferenceConfig(ensemble="invalid")


def test_inference_config_invalid_values():
    with pytest.raises(ValidationError):
        InferenceConfig(temperature=-10.0)
    with pytest.raises(ValidationError):
        InferenceConfig(timestep=0.0)
    with pytest.raises(ValidationError):
        InferenceConfig(zbl_inner_cutoff=-1.0)


def test_inference_config_executable_validation():
    # Valid path string is okay
    config = InferenceConfig(lammps_executable="/usr/bin/lmp")
    assert str(config.lammps_executable) == "/usr/bin/lmp"
