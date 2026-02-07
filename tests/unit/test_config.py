import pytest
from pydantic import ValidationError
from mlip_autopipec.domain_models.config import GlobalConfig

def test_valid_config() -> None:
    data = {
        "project_name": "test_project",
        "seed": 42,
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "generator": {"type": "mock"},
    }
    config = GlobalConfig(**data)
    assert config.project_name == "test_project"
    assert config.seed == 42
    assert config.oracle.type == "mock"

def test_invalid_config_missing_field() -> None:
    data = {
        "project_name": "test_project",
        # Missing seed
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "generator": {"type": "mock"},
    }
    with pytest.raises(ValidationError):
        GlobalConfig(**data)

def test_invalid_config_wrong_type() -> None:
    data = {
        "project_name": "test_project",
        "seed": 42,
        "oracle": {"type": "invalid_type"}, # Invalid type
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "generator": {"type": "mock"},
    }
    with pytest.raises(ValidationError):
        GlobalConfig(**data)
