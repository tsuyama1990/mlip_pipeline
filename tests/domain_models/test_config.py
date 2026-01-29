"""Tests for configuration domain models."""

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import Config, PotentialConfig


def test_config_valid(temp_dir: object) -> None:
    """Test creating a valid configuration."""
    data = {
        "project_name": "test_project",
        "potential": {
            "elements": ["Si"],
            "cutoff": 5.0,
            "seed": 123
        },
        "logging": {
            "level": "DEBUG",
            "file_path": "test.log"
        }
    }
    config = Config(**data)  # type: ignore[arg-type]
    assert config.project_name == "test_project"
    assert config.potential.cutoff == 5.0


def test_config_invalid_cutoff() -> None:
    """Test invalid cutoff."""
    data = {
        "project_name": "test_project",
        "potential": {
            "elements": ["Si"],
            "cutoff": -1.0,  # Invalid
            "seed": 123
        }
    }
    with pytest.raises(ValidationError) as excinfo:
        Config(**data)  # type: ignore[arg-type]
    assert "Cutoff must be greater than 0" in str(excinfo.value)


def test_config_invalid_elements() -> None:
    """Test empty elements list."""
    with pytest.raises(ValidationError):
        PotentialConfig(elements=[], cutoff=5.0)


def test_config_invalid_project_name() -> None:
    """Test empty project name."""
    with pytest.raises(ValidationError):
        Config(
            project_name="  ",
            potential=PotentialConfig(elements=["Si"], cutoff=5.0)
        )
