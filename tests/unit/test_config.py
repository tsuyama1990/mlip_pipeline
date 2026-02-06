import pytest
from pathlib import Path
from pydantic import ValidationError
from mlip_autopipec.config.config_model import GlobalConfig

def test_valid_config() -> None:
    config_data = {
        "work_dir": "/tmp/test_work_dir",
        "max_cycles": 5,
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "explorer": {"type": "mock"}
    }
    config = GlobalConfig(**config_data)
    assert config.work_dir == Path("/tmp/test_work_dir")
    assert config.max_cycles == 5
    assert config.oracle.type == "mock"

def test_invalid_config_missing_field() -> None:
    config_data = {
        "max_cycles": 5
    }
    with pytest.raises(ValidationError):
        GlobalConfig(**config_data) # type: ignore[call-arg]

def test_invalid_config_type_mismatch() -> None:
    config_data = {
        "work_dir": "/tmp/test_work_dir",
        "oracle": {"type": "invalid_type"}
    }
    with pytest.raises(ValidationError):
        GlobalConfig(**config_data)

def test_defaults() -> None:
    config_data = {
        "work_dir": "/tmp/test_work_dir"
    }
    config = GlobalConfig(**config_data)
    assert config.max_cycles == 10
    assert config.oracle.type == "mock"
    assert config.trainer.type == "mock"
    assert config.explorer.type == "mock"
