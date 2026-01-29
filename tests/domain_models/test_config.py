from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import Config, PotentialConfig


def test_config_valid() -> None:
    """Test valid configuration creation."""
    c = Config(
        project_name="TestProject",
        potential=PotentialConfig(
            elements=["Ti", "O"],
            cutoff=5.0,
            seed=42
        )
    )
    assert c.project_name == "TestProject"
    assert c.potential.cutoff == 5.0
    assert c.logging.level == "INFO" # Default

def test_config_invalid_cutoff() -> None:
    """Test negative cutoff."""
    with pytest.raises(ValidationError) as excinfo:
        Config(
            project_name="Test",
            potential=PotentialConfig(
                elements=["Ti"],
                cutoff=-1.0, # Invalid
                seed=42
            )
        )
    assert "Cutoff must be greater than 0" in str(excinfo.value)

def test_config_missing_field() -> None:
    """Test missing required field."""
    with pytest.raises(ValidationError):
        Config(
            project_name="Test",
            potential=None,
            # Missing potential
        )

def test_from_yaml(tmp_path: Path) -> None:
    """Test loading from YAML using real IO."""
    yaml_file = tmp_path / "config.yaml"
    yaml_content = """
    project_name: "YamlProject"
    potential:
      elements: ["Cu"]
      cutoff: 3.0
      seed: 123
    logging:
      level: "DEBUG"
    """
    yaml_file.write_text(yaml_content)

    c = Config.from_yaml(yaml_file)

    assert c.project_name == "YamlProject"
    assert c.potential.elements == ["Cu"]
    assert c.logging.level == "DEBUG"


def test_full_config_valid() -> None:
    """Test full configuration with all sub-configs."""
    c = Config(
        project_name="FullProject",
        potential=PotentialConfig(elements=["Al"], cutoff=4.0),
        exploration={"max_steps": 500, "temperature": 500.0},
        dft={"command": "vasp_std", "kspacing": 0.05},
        training={"max_epochs": 10},
        validation={"check_phonons": False}
    )

    assert c.exploration.max_steps == 500
    assert c.exploration.temperature == 500.0
    assert c.dft.command == "vasp_std"
    assert c.training.max_epochs == 10
    assert not c.validation.check_phonons
    # Check default
    assert c.exploration.time_step == 0.001
