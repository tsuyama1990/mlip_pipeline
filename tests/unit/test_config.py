from typing import Any

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models import GlobalConfig


def test_valid_config() -> None:
    data: dict[str, Any] = {
        "project_name": "test_project",
        "workdir": "/tmp/test_project",  # noqa: S108
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
    assert config.oracle.type == "mock"
    assert str(config.workdir) == "/tmp/test_project" # noqa: S108


def test_missing_oracle() -> None:
    data: dict[str, Any] = {
        "project_name": "test_project",
        "workdir": "/tmp/test_project", # noqa: S108
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "generator": {"type": "mock"},
    }
    with pytest.raises(ValidationError):
        GlobalConfig(**data)
