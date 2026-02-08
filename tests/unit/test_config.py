from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import GlobalConfig


def test_config_valid(tmp_path: Path) -> None:
    # Minimal valid config dictionary
    config_dict: dict[str, Any] = {
        "workdir": tmp_path,
        "max_cycles": 10,
        "logging_level": "INFO",
        "components": {
            "generator": {"name": "mock"},
            "oracle": {"name": "mock"},
            "trainer": {"name": "mock"},
            "dynamics": {"name": "mock"},
            "validator": {"name": "mock"},
        }
    }

    config = GlobalConfig.model_validate(config_dict)
    assert config.max_cycles == 10
    assert config.components.generator.name == "mock"


def test_config_missing_components(tmp_path: Path) -> None:
    # Pydantic will raise ValidationError for missing fields in ComponentsConfig
    with pytest.raises(ValidationError) as excinfo:
        GlobalConfig.model_validate({
            "workdir": tmp_path,
            "max_cycles": 10,
            "components": {"generator": {"name": "mock"}},  # Missing others
        })

    # Check that error mentions missing fields (e.g. 'oracle', 'trainer', etc.)
    assert "Field required" in str(excinfo.value)
    assert "oracle" in str(excinfo.value)


def test_config_extra_forbidden(tmp_path: Path) -> None:
    components: dict[str, dict[str, Any]] = {
        "generator": {"name": "mock", "extra_field": 123},
        "oracle": {"name": "mock"},
        "trainer": {"name": "mock"},
        "dynamics": {"name": "mock"},
        "validator": {"name": "mock"},
    }

    with pytest.raises(ValidationError) as excinfo:
        GlobalConfig.model_validate({
            "workdir": tmp_path,
            "max_cycles": 10,
            "components": components
        })
    assert "Extra inputs are not permitted" in str(excinfo.value)
