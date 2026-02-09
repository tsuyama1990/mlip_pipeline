from pathlib import Path

import pytest

from mlip_autopipec.config import ExperimentConfig
from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models.enums import OracleType


def test_orchestrator_init_with_config(tmp_path: Path) -> None:
    """Test initializing Orchestrator with an ExperimentConfig object."""
    config = ExperimentConfig()
    config.orchestrator.work_dir = tmp_path / "work"

    orchestrator = Orchestrator(config)

    assert orchestrator.config.orchestrator.work_dir == tmp_path / "work"
    assert orchestrator.logger.name == "Orchestrator"

    # Check if components are instantiated
    assert orchestrator.oracle is not None
    assert orchestrator.generator is not None


def test_orchestrator_init_with_path(tmp_path: Path) -> None:
    """Test initializing Orchestrator with a path to YAML file."""
    config_content = """
    orchestrator:
      work_dir: "orch_test"
    oracle:
      type: "MOCK"
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)

    orchestrator = Orchestrator(config_file)

    assert orchestrator.config.oracle.type == OracleType.MOCK
    assert orchestrator.config.orchestrator.work_dir == Path("orch_test")


def test_orchestrator_init_invalid_type() -> None:
    """Test initializing Orchestrator with invalid type."""
    with pytest.raises(TypeError, match="Invalid config type"):
        Orchestrator(123)  # type: ignore


def test_orchestrator_component_init_failure(tmp_path: Path) -> None:
    """Test handling of component initialization failure."""
    config = ExperimentConfig()
    config.orchestrator.work_dir = tmp_path / "work_fail"

    from unittest.mock import patch

    with (
        patch(
            "mlip_autopipec.core.orchestrator.ComponentFactory.create",
            side_effect=RuntimeError("Factory Error"),
        ),
        pytest.raises(RuntimeError, match="Failed to initialize components"),
    ):
        Orchestrator(config)
