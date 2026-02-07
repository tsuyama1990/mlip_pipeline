from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models import GlobalConfig


def test_config_valid() -> None:
    data: dict[str, Any] = {
        "workdir": "workdir",
        "max_cycles": 10,
        "oracle": {"type": "mock", "params": {}},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "generator": {"type": "mock"},
        "validator": {"type": "mock"},
        "selector": {"type": "mock"},
    }
    config = GlobalConfig(**data)
    assert config.workdir == Path("workdir")
    assert config.oracle.type == "mock"


def test_config_invalid_type() -> None:
    data: dict[str, Any] = {
        "workdir": "workdir",
        "max_cycles": 10,
        "oracle": {"type": "invalid_type"},  # Invalid Literal
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "generator": {"type": "mock"},
        "validator": {"type": "mock"},
        "selector": {"type": "mock"},
    }
    with pytest.raises(ValidationError):
        GlobalConfig(**data)
