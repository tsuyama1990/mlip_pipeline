import pytest
from pathlib import Path
from typing import Any, Dict
from pydantic import ValidationError
from mlip_autopipec.config.config_model import GlobalConfig

def test_valid_global_config(tmp_path: Path) -> None:
    config_data: Dict[str, Any] = {
        "work_dir": tmp_path / "test",
        "max_cycles": 5,
        "random_seed": 123,
    }
    config = GlobalConfig(**config_data)
    assert config.max_cycles == 5
    assert config.explorer.type == "mock"

def test_invalid_cycles(tmp_path: Path) -> None:
    config_data: Dict[str, Any] = {
        "work_dir": tmp_path / "test",
        "max_cycles": 0,
        "random_seed": 123,
    }
    with pytest.raises(ValidationError):
        GlobalConfig(**config_data)

def test_extra_forbidden(tmp_path: Path) -> None:
    config_data: Dict[str, Any] = {
        "work_dir": tmp_path / "test",
        "max_cycles": 5,
        "extra_field": "value",
    }
    with pytest.raises(ValidationError):
        GlobalConfig(**config_data)
