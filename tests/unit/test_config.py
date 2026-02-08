from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models import GlobalConfig


def test_valid_config() -> None:
    """Test creating a valid GlobalConfig."""
    config_dict = {
        "workdir": "runs/test",
        "max_cycles": 1,
        "generator": {"type": "mock", "extra_param": 123},
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "validator": {"type": "mock"},
    }
    config = GlobalConfig(**config_dict)
    assert config.workdir == Path("runs/test")
    assert config.max_cycles == 1
    assert config.generator.type == "mock"
    assert getattr(config.generator, "extra_param", None) == 123


def test_missing_required_fields() -> None:
    """Test validation error for missing required fields."""
    with pytest.raises(ValidationError) as excinfo:
        GlobalConfig(max_cycles=1)  # Missing workdir and components
    assert "Field required" in str(excinfo.value)


def test_invalid_field_type() -> None:
    """Test validation error for invalid field types."""
    with pytest.raises(ValidationError):
        GlobalConfig(
            workdir="runs/test",
            max_cycles="invalid_int",  # Should be int
            generator={"type": "mock"},
            oracle={"type": "mock"},
            trainer={"type": "mock"},
            dynamics={"type": "mock"},
        )


def test_missing_component_type() -> None:
    """Test validation error if component config is missing 'type'."""
    with pytest.raises(ValidationError):
        GlobalConfig(
            workdir="runs/test",
            max_cycles=1,
            generator={},  # Missing type
            oracle={"type": "mock"},
            trainer={"type": "mock"},
            dynamics={"type": "mock"},
        )
