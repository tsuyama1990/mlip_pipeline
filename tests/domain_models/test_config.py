from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import Config, PotentialConfig, StructureGenConfig, LammpsConfig


def test_config_valid() -> None:
    """Test valid configuration creation."""
    c = Config(
        project_name="TestProject",
        potential=PotentialConfig(
            elements=["Ti", "O"],
            cutoff=5.0,
            seed=42
        ),
        structure_gen=StructureGenConfig(
            element="Ti",
            crystal_structure="hcp",
            lattice_constant=2.95
        )
    )
    assert c.project_name == "TestProject"
    assert c.potential.cutoff == 5.0
    assert c.logging.level == "INFO" # Default
    assert c.structure_gen.element == "Ti"
    assert c.lammps.command == "lmp_serial" # Default

def test_config_invalid_cutoff() -> None:
    """Test negative cutoff."""
    with pytest.raises(ValidationError) as excinfo:
        Config(
            project_name="Test",
            potential=PotentialConfig(
                elements=["Ti"],
                cutoff=-1.0, # Invalid
                seed=42
            ),
            structure_gen=StructureGenConfig(
                element="Ti",
                crystal_structure="hcp",
                lattice_constant=2.95
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
            structure_gen=StructureGenConfig(
                element="Ti",
                crystal_structure="hcp",
                lattice_constant=2.95
            )
        )

def test_structure_gen_defaults() -> None:
    """Test defaults for StructureGenConfig."""
    sg = StructureGenConfig(element="Si", crystal_structure="diamond", lattice_constant=5.43)
    assert sg.supercell == (3, 3, 3)

def test_lammps_config_explicit() -> None:
    """Test explicit LammpsConfig creation."""
    lc = LammpsConfig(command="lmp_mpi", timeout=10.0)
    assert lc.command == "lmp_mpi"
    assert lc.timeout == 10.0

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
    structure_gen:
      element: "Cu"
      crystal_structure: "fcc"
      lattice_constant: 3.6
      supercell: [2, 2, 2]
    lammps:
      command: "lmp_mpi"
      timeout: 100.0
    """
    yaml_file.write_text(yaml_content)

    c = Config.from_yaml(yaml_file)

    assert c.project_name == "YamlProject"
    assert c.potential.elements == ["Cu"]
    assert c.logging.level == "DEBUG"
    assert c.structure_gen.supercell == (2, 2, 2)
    assert c.lammps.command == "lmp_mpi"
    assert c.lammps.timeout == 100.0
