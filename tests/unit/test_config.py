from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import (
    GlobalConfig,
    MockDynamicsConfig,
    MockGeneratorConfig,
    MockOracleConfig,
    MockSelectorConfig,
    MockTrainerConfig,
    MockValidatorConfig,
)


def test_config_valid(tmp_path: Path) -> None:
    # Create valid minimal config using dict input
    config_dict: dict[str, Any] = {
        "workdir": tmp_path,
        "max_cycles": 10,
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "generator": {"type": "mock"},
        "validator": {"type": "mock"},
        "selector": {"type": "mock"},
    }
    config = GlobalConfig.model_validate(config_dict)

    assert config.workdir == tmp_path
    assert config.max_cycles == 10
    assert isinstance(config.oracle, MockOracleConfig)
    assert isinstance(config.trainer, MockTrainerConfig)
    assert isinstance(config.dynamics, MockDynamicsConfig)
    assert isinstance(config.generator, MockGeneratorConfig)
    assert isinstance(config.validator, MockValidatorConfig)
    assert isinstance(config.selector, MockSelectorConfig)


def test_config_validation_errors(tmp_path: Path) -> None:
    # Max cycles invalid
    config_dict: dict[str, Any] = {
        "workdir": tmp_path,
        "max_cycles": 0,
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "generator": {"type": "mock"},
        "validator": {"type": "mock"},
        "selector": {"type": "mock"},
    }
    with pytest.raises(ValidationError) as exc:
        GlobalConfig.model_validate(config_dict)
    assert "max_cycles must be positive" in str(exc.value)


def test_config_defaults(tmp_path: Path) -> None:
    # Minimal config should use defaults
    config_dict: dict[str, Any] = {
        "workdir": tmp_path,
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "generator": {"type": "mock"},
        "validator": {"type": "mock"},
        "selector": {"type": "mock"},
    }
    config = GlobalConfig.model_validate(config_dict)
    assert config.max_cycles == 10
    assert config.initial_structure_path is None


def test_discriminator_loading(tmp_path: Path) -> None:
    # Test that discriminator correctly selects model based on 'type'
    config_dict: dict[str, Any] = {
        "workdir": tmp_path,
        "oracle": {"type": "mock", "params": {"key": "val"}},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock", "halt_probability": 0.8},
        "generator": {"type": "mock", "n_candidates": 10},
        "validator": {"type": "mock"},
        "selector": {"type": "mock"},
    }

    config = GlobalConfig.model_validate(config_dict)
    assert isinstance(config.oracle, MockOracleConfig)
    assert config.oracle.params["key"] == "val"
    assert config.dynamics.halt_probability == 0.8
    assert config.generator.n_candidates == 10
