from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import Config, PotentialConfig, LammpsConfig


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
    assert c.lammps.command == "lmp_serial" # Default

def test_config_with_lammps() -> None:
    c = Config(
        project_name="TestProject",
        potential=PotentialConfig(elements=["Si"], cutoff=4.0),
        lammps=LammpsConfig(command="mpirun lmp", cores=4, timeout=10.0)
    )
    assert c.lammps.command == "mpirun lmp"
    assert c.lammps.cores == 4
    assert c.lammps.timeout == 10.0

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
            potential=None, # type: ignore[arg-type]
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
    lammps:
      command: "lmp"
      cores: 2
    """
    yaml_file.write_text(yaml_content)

    c = Config.from_yaml(yaml_file)

    assert c.project_name == "YamlProject"
    assert c.potential.elements == ["Cu"]
    assert c.logging.level == "DEBUG"
    assert c.lammps.command == "lmp"
    assert c.lammps.cores == 2
