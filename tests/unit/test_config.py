from typing import Any

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import GlobalConfig


def test_valid_config() -> None:
    data: dict[str, Any] = {
        "project_name": "test_project",
        "seed": 42,
        "max_cycles": 5,
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "generator": {"type": "mock"},
        "validator": {"type": "mock"},
        "selector": {"type": "mock"},
    }
    config = GlobalConfig(**data)
    assert config.project_name == "test_project"
    assert config.seed == 42
    assert config.max_cycles == 5
    assert config.oracle.type == "mock"
    assert config.oracle.params == {}
    assert config.validator.type == "mock"
    assert config.selector.type == "mock"

def test_invalid_config_missing_field() -> None:
    data: dict[str, Any] = {
        "project_name": "test_project",
        # Missing seed
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "generator": {"type": "mock"},
        "validator": {"type": "mock"},
        "selector": {"type": "mock"},
    }
    with pytest.raises(ValidationError):
        GlobalConfig(**data)

def test_invalid_config_wrong_type() -> None:
    data: dict[str, Any] = {
        "project_name": "test_project",
        "seed": 42,
        "oracle": {"type": "invalid_type"}, # Invalid type
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "generator": {"type": "mock"},
        "validator": {"type": "mock"},
        "selector": {"type": "mock"},
    }
    with pytest.raises(ValidationError):
        GlobalConfig(**data)
