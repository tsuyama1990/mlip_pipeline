from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import (
    Config,
    LammpsConfig,
    PotentialConfig,
    StructureGenConfig,
)


def test_config_valid() -> None:
    """Test valid configuration creation."""
    c = Config(
        project_name="TestProject",
        potential=PotentialConfig(elements=["Ti", "O"], cutoff=5.0, seed=42),
    )
    assert c.project_name == "TestProject"
    assert c.potential.cutoff == 5.0
    assert c.logging.level == "INFO"  # Default
    assert isinstance(c.lammps, LammpsConfig)
    assert isinstance(c.structure_gen, StructureGenConfig)


def test_config_invalid_cutoff() -> None:
    """Test negative cutoff."""
    with pytest.raises(ValidationError) as excinfo:
        Config(
            project_name="Test",
            potential=PotentialConfig(
                elements=["Ti"],
                cutoff=-1.0,  # Invalid
                seed=42,
            ),
        )
    assert "Cutoff must be greater than 0" in str(excinfo.value)


def test_config_missing_field() -> None:
    """Test missing required field."""
    with pytest.raises(ValidationError):
        Config(
            project_name="Test",
            potential=None,  # type: ignore[arg-type]
            # Missing potential
        )


def test_lammps_config_defaults() -> None:
    """Test LammpsConfig defaults."""
    c = LammpsConfig()
    assert c.command == "lmp_serial"
    assert c.timeout == 3600
    assert c.use_mpi is False


def test_structure_gen_config_defaults() -> None:
    """Test StructureGenConfig defaults."""
    c = StructureGenConfig()
    assert c.element == "Si"
    assert c.rattle_stdev == 0.0
    assert c.supercell == (2, 2, 2)


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


def test_from_yaml_full(tmp_path: Path) -> None:
    """Test loading full configuration from YAML."""
    yaml_file = tmp_path / "config_full.yaml"
    yaml_content = """
    project_name: "FullProject"
    potential:
      elements: ["Cu"]
      cutoff: 3.0
    lammps:
      command: "lmp_mpi"
      use_mpi: true
      timeout: 10
    structure_gen:
      element: "Al"
      lattice_constant: 4.05
    """
    yaml_file.write_text(yaml_content)

    c = Config.from_yaml(yaml_file)
    assert c.lammps.command == "lmp_mpi"
    assert c.lammps.use_mpi is True
    assert c.structure_gen.element == "Al"
    assert c.structure_gen.lattice_constant == 4.05
