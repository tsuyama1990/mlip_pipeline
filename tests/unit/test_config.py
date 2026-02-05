from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mlip_autopipec.config.config_model import GlobalConfig


def test_load_valid_config(tmp_path: Path) -> None:
    config_dict = {
        "work_dir": str(tmp_path),
        "logging_level": "DEBUG",
        "exploration": {"strategy": "md", "max_structures": 5}
    }
    # Pydantic can load from dict
    config = GlobalConfig(**config_dict) # type: ignore[arg-type]
    assert config.logging_level == "DEBUG"
    assert config.exploration.strategy == "md"
    assert config.dft.calculator == "mock" # Default

def test_load_invalid_extra_field(tmp_path: Path) -> None:
    config_dict = {
        "work_dir": str(tmp_path),
        "extra_field": "fail"
    }
    with pytest.raises(ValidationError):
        GlobalConfig(**config_dict) # type: ignore[arg-type]

def test_load_from_yaml(tmp_path: Path) -> None:
    config_data = {
        "work_dir": str(tmp_path),
        "logging_level": "INFO"
    }
    config_file = tmp_path / "config.yaml"
    with config_file.open("w") as f:
        yaml.dump(config_data, f)

    with config_file.open("r") as f:
        loaded_data = yaml.safe_load(f)

    config = GlobalConfig(**loaded_data)
    assert config.work_dir == tmp_path
