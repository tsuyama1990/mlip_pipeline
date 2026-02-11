from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models import GlobalConfig


def test_global_config_valid() -> None:
    config_dict = {"orchestrator": {"work_dir": "test_run", "max_cycles": 5}}
    config = GlobalConfig.model_validate(config_dict)
    assert config.orchestrator.work_dir == Path("test_run")
    assert config.orchestrator.max_cycles == 5


def test_global_config_invalid_missing_field() -> None:
    config_dict = {"orchestrator": {"max_cycles": 5}}
    with pytest.raises(ValidationError):
        GlobalConfig.model_validate(config_dict)


def test_global_config_invalid_type() -> None:
    config_dict = {"orchestrator": {"work_dir": "test_run", "max_cycles": "invalid"}}
    with pytest.raises(ValidationError):
        GlobalConfig.model_validate(config_dict)


def test_global_config_extra_field() -> None:
    config_dict = {"orchestrator": {"work_dir": "test_run", "extra_field": "invalid"}}
    with pytest.raises(ValidationError):
        GlobalConfig.model_validate(config_dict)
