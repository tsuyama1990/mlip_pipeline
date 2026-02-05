import pytest
from pydantic import ValidationError
from config import GlobalConfig
from pathlib import Path

def test_global_config_valid() -> None:
    config = GlobalConfig(work_dir=Path("./tmp"), max_cycles=10, random_seed=123)
    assert config.max_cycles == 10
    assert config.random_seed == 123

def test_global_config_invalid_max_cycles() -> None:
    with pytest.raises(ValidationError):
        GlobalConfig(work_dir=Path("./tmp"), max_cycles=0, random_seed=42)  # Must be > 0

def test_global_config_invalid_types() -> None:
    with pytest.raises(ValidationError):
        GlobalConfig(work_dir=Path("./tmp"), max_cycles="ten", random_seed=42)
