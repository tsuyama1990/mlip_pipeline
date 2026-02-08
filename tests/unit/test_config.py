from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models import GlobalConfig


def test_valid_config() -> None:
    """Test creating a valid GlobalConfig."""
    config_dict: Any = {
        "workdir": "runs/test",
        "max_cycles": 1,
        "generator": {"type": "mock", "params": {"extra_param": 123}},
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "validator": {"type": "mock"},
    }
    config = GlobalConfig(**config_dict)
    assert config.workdir == Path("runs/test")
    assert config.max_cycles == 1
    assert config.generator.type == "mock"
    assert config.generator.params["extra_param"] == 123


def test_config_defaults() -> None:
    """Test default values."""
    config_dict: Any = {
        # workdir omitted -> default runs/default
        # max_cycles omitted -> default 5
        "generator": {"type": "mock"},
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
    }
    config = GlobalConfig(**config_dict)
    assert config.workdir == Path("runs/default")
    assert config.max_cycles == 5


def test_invalid_field_type() -> None:
    """Test validation error for invalid field types."""
    with pytest.raises(ValidationError):
        GlobalConfig(
            workdir="runs/test",  # type: ignore
            max_cycles="invalid_int",  # type: ignore
            generator={"type": "mock"},  # type: ignore
            oracle={"type": "mock"},  # type: ignore
            trainer={"type": "mock"},  # type: ignore
            dynamics={"type": "mock"},  # type: ignore
        )


def test_dataset_path_security() -> None:
    """Test validation of dataset_filename path traversal."""
    with pytest.raises(ValidationError, match="dataset_filename cannot contain parent directory traversal"):
        GlobalConfig(
            dataset_filename="../bad.jsonl",
            generator={"type": "mock"},  # type: ignore
            oracle={"type": "mock"},  # type: ignore
            trainer={"type": "mock"},  # type: ignore
            dynamics={"type": "mock"},  # type: ignore
        )

    with pytest.raises(ValidationError, match="dataset_filename must be relative"):
        GlobalConfig(
            dataset_filename="/absolute/path.jsonl",
            generator={"type": "mock"},  # type: ignore
            oracle={"type": "mock"},  # type: ignore
            trainer={"type": "mock"},  # type: ignore
            dynamics={"type": "mock"},  # type: ignore
        )
