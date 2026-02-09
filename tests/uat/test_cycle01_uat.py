from pathlib import Path

import pytest

from mlip_autopipec.config import ExperimentConfig
from mlip_autopipec.domain_models.enums import OracleType


def test_uat_config_loading(tmp_path: Path) -> None:
    """UAT Scenario 1.1: Configuration Loading."""
    config_content = """
    orchestrator:
      work_dir: "uat_work"
    oracle:
      type: "MOCK"
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)

    config = ExperimentConfig.from_yaml(config_file)
    assert isinstance(config, ExperimentConfig)
    assert config.oracle.type == OracleType.MOCK


def test_uat_invalid_config(tmp_path: Path) -> None:
    """UAT Scenario 1.2: Invalid Configuration Handling."""
    config_content = """
    oracle:
      type: "INVALID_TYPE"
    """
    config_file = tmp_path / "bad_config.yaml"
    config_file.write_text(config_content)

    with pytest.raises(ValueError, match="validation error"):
        ExperimentConfig.from_yaml(config_file)


def test_uat_component_instantiation(tmp_path: Path) -> None:
    """UAT Scenario 1.3: Component Instantiation via Factory."""
    # This test requires Orchestrator implementation
    config_content = """
    orchestrator:
      work_dir: "uat_work"
    oracle:
      type: "MOCK"
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)

    # We will instantiate Orchestrator with this config in the future
