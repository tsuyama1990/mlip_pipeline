from pathlib import Path

import pytest
from pydantic import ValidationError

from config import GlobalConfig


def test_global_config_valid():
    config = GlobalConfig(work_dir=Path("./tmp"), max_cycles=10, random_seed=123)
    assert config.max_cycles == 10
    assert config.random_seed == 123


def test_global_config_invalid_max_cycles():
    with pytest.raises(ValidationError):
        GlobalConfig(work_dir=Path("./tmp"), max_cycles=0)  # Must be > 0


def test_global_config_invalid_types():
    with pytest.raises(ValidationError):
        GlobalConfig(work_dir=Path("./tmp"), max_cycles="ten")
