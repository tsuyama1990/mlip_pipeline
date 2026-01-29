from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import Config, PotentialConfig


def test_potential_config_validation() -> None:
    with pytest.raises(ValidationError):
        PotentialConfig(elements=["H"], cutoff=-1.0)  # Invalid cutoff


def test_config_structure() -> None:
    # Minimal valid config
    config = Config(
        project_name="test_project",
        potential=PotentialConfig(elements=["H"], cutoff=5.0)
    )
    assert config.project_name == "test_project"
    assert config.logging.level == "INFO"
    assert config.structure_gen.enabled is False
    assert config.oracle.enabled is False
    assert config.trainer.enabled is False
    assert config.dynamics.enabled is False
    assert config.orchestrator is not None


def test_config_from_yaml_mocked(tmp_path: Path) -> None:
    with patch("mlip_autopipec.infrastructure.io.load_yaml") as mock_load:
        mock_load.return_value = {
            "project_name": "yaml_test",
            "potential": {"elements": ["Si"], "cutoff": 4.0},
            "logging": {"level": "DEBUG"}
        }

        config = Config.from_yaml(tmp_path / "config.yaml")

        assert config.project_name == "yaml_test"
        assert config.logging.level == "DEBUG"
        mock_load.assert_called_once()
