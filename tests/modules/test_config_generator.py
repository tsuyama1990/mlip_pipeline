# ruff: noqa: D101, D102, D103, D107
"""Tests for the PacemakerConfigGenerator module."""

from pathlib import Path

import pytest
import yaml

from mlip_autopipec.config.pacemaker_config import PacemakerConfig
from mlip_autopipec.config_schemas import SystemConfig
from mlip_autopipec.modules.config_generator import PacemakerConfigGenerator


@pytest.fixture
def test_system_config(tmp_path: Path) -> SystemConfig:
    """Provide a default SystemConfig for testing."""
    dft_config = {"executable": {}, "input": {"pseudopotentials": {"Ni": "ni.upf"}}}
    config = SystemConfig(dft=dft_config, db_path=str(tmp_path / "test.db"))
    config.trainer.loss_weights.energy = 2.0
    config.trainer.ace_params.correlation_order = 4
    return config


def test_generate_pacemaker_config(
    test_system_config: SystemConfig, tmp_path: Path
) -> None:
    """Unit test for the Pacemaker config generation logic."""
    generator = PacemakerConfigGenerator(test_system_config)
    dummy_data_path = tmp_path / "dummy_data.xyz"
    dummy_data_path.touch()

    config_path = generator.generate_config(dummy_data_path, tmp_path)
    assert config_path.exists()

    # Verify that the generated YAML can be loaded and parsed by the Pydantic model
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
        parsed_config = PacemakerConfig(**config_data)

    # Verify that the generated config matches the SystemConfig
    fit_params = parsed_config.fit_params
    assert fit_params.dataset_filename == str(dummy_data_path)
    assert (
        fit_params.loss_weights.energy == test_system_config.trainer.loss_weights.energy
    )
    assert (
        fit_params.ace.correlation_order
        == test_system_config.trainer.ace_params.correlation_order
    )
