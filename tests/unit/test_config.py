from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.config import ExperimentConfig
from mlip_autopipec.domain_models.enums import OracleType


def test_default_config() -> None:
    """Test that default configuration is valid."""
    config = ExperimentConfig()
    assert config.oracle.type == OracleType.MOCK
    assert config.orchestrator.max_cycles == 10


def test_load_valid_yaml(tmp_path: Path) -> None:
    """Test loading a valid YAML configuration."""
    config_content = """
    orchestrator:
      max_cycles: 5
      work_dir: "my_experiment"
    oracle:
      type: "QE"
      scf_kspacing: 0.4
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)

    config = ExperimentConfig.from_yaml(config_file)
    assert config.orchestrator.max_cycles == 5
    assert config.oracle.type == OracleType.QE
    assert config.oracle.scf_kspacing == 0.4


def test_load_invalid_yaml_missing_field(tmp_path: Path) -> None:
    """Test that invalid YAML raises ValidationError (wrong type)."""
    config_content = """
    orchestrator:
      max_cycles: "not_an_int"
    """
    config_file = tmp_path / "bad_config.yaml"
    config_file.write_text(config_content)

    with pytest.raises(ValidationError):
        ExperimentConfig.from_yaml(config_file)


def test_load_invalid_yaml_extra_field(tmp_path: Path) -> None:
    """Test that extra fields are forbidden."""
    config_content = """
    orchestrator:
      unknown_field: "value"
    """
    config_file = tmp_path / "bad_config_extra.yaml"
    config_file.write_text(config_content)

    with pytest.raises(ValidationError):
        ExperimentConfig.from_yaml(config_file)


def test_file_not_found() -> None:
    """Test that FileNotFoundError is raised for missing config."""
    with pytest.raises(FileNotFoundError):
        ExperimentConfig.from_yaml("non_existent_config.yaml")


def test_path_traversal_validation(tmp_path: Path) -> None:
    """Test that path traversal (..) is detected and rejected."""
    # Test work_dir validation
    config_content = """
    orchestrator:
      work_dir: "../unsafe_path"
    """
    config_file = tmp_path / "unsafe_config.yaml"
    config_file.write_text(config_content)

    with pytest.raises(ValidationError, match="Path traversal detected"):
        ExperimentConfig.from_yaml(config_file)

    # Test pseudopotentials validation
    config_content_pseudo = """
    oracle:
      pseudopotentials:
        Fe: "../Fe.upf"
    """
    config_file_pseudo = tmp_path / "unsafe_pseudo.yaml"
    config_file_pseudo.write_text(config_content_pseudo)

    with pytest.raises(ValidationError, match="Path traversal detected"):
        ExperimentConfig.from_yaml(config_file_pseudo)


def test_restricted_system_path_validation(tmp_path: Path) -> None:
    """Test that restricted system paths are rejected."""
    config_content = """
    orchestrator:
      work_dir: "/etc/passwd"
    """
    config_file = tmp_path / "system_path_config.yaml"
    config_file.write_text(config_content)

    with pytest.raises(ValidationError, match="Path points to restricted system directory"):
        ExperimentConfig.from_yaml(config_file)
