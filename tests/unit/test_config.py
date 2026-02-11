from typing import Any

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models import GeneratorType, GlobalConfig


def test_valid_config(valid_config_dict: dict[str, Any]) -> None:
    config = GlobalConfig(**valid_config_dict)
    assert config.project_name == "TestProject"
    assert config.generator.type == GeneratorType.MOCK
    assert config.orchestrator.max_iterations == 10

def test_invalid_iterations(valid_config_dict: dict[str, Any]) -> None:
    valid_config_dict["orchestrator"]["max_iterations"] = -1
    with pytest.raises(ValidationError):
        GlobalConfig(**valid_config_dict)

def test_missing_field(valid_config_dict: dict[str, Any]) -> None:
    del valid_config_dict["orchestrator"]
    with pytest.raises(ValidationError):
        GlobalConfig(**valid_config_dict)

def test_invalid_enum(valid_config_dict: dict[str, Any]) -> None:
    valid_config_dict["generator"]["type"] = "INVALID_TYPE"
    with pytest.raises(ValidationError):
        GlobalConfig(**valid_config_dict)
