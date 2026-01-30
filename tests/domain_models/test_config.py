from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import Config, PotentialConfig, LammpsConfig, StructureGenConfig


def test_config_valid() -> None:
    """Test valid configuration creation."""
    c = Config(
        project_name="TestProject",
        potential=PotentialConfig(
            elements=["Ti", "O"],
            cutoff=5.0,
            seed=42
        ),
        lammps=LammpsConfig(command="lmp"),
        structure_gen=StructureGenConfig(
            element="Ti",
            crystal_structure="hcp",
            lattice_constant=2.95,
            supercell=(1, 1, 1)
        )
    )
    assert c.project_name == "TestProject"
    assert c.potential.cutoff == 5.0
    assert c.logging.level == "INFO" # Default
    assert c.lammps.command == "lmp"

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
            lammps=LammpsConfig(command="lmp"),
            structure_gen=StructureGenConfig(
                element="Ti",
                crystal_structure="hcp",
                lattice_constant=2.95,
                supercell=(1, 1, 1)
            )
        )
    assert "Cutoff must be greater than 0" in str(excinfo.value)

def test_config_missing_field() -> None:
    """Test missing required field."""
    with pytest.raises(ValidationError):
        Config(
            project_name="Test",
            potential=None, # type: ignore[arg-type]
            lammps=LammpsConfig(command="echo"),
            structure_gen=StructureGenConfig(
                element="Si",
                crystal_structure="diamond",
                lattice_constant=5.43,
                supercell=(1, 1, 1)
            )
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
      command: "lmp_serial"
    structure_gen:
      element: "Cu"
      crystal_structure: "fcc"
      lattice_constant: 3.61
      supercell: [2, 2, 2]
    """
    yaml_file.write_text(yaml_content)

    c = Config.from_yaml(yaml_file)

    assert c.project_name == "YamlProject"
    assert c.potential.elements == ["Cu"]
    assert c.logging.level == "DEBUG"
    assert c.lammps.command == "lmp_serial"
    assert c.structure_gen.supercell == (2, 2, 2)

def test_lammps_config_defaults() -> None:
    """Test LammpsConfig defaults."""
    c = LammpsConfig(command="lmp")
    assert c.command == "lmp"
    assert c.cores == 1
    assert c.timeout == 3600.0

def test_structure_gen_config() -> None:
    """Test StructureGenConfig."""
    c = StructureGenConfig(
        element="Si",
        crystal_structure="diamond",
        lattice_constant=5.43,
        supercell=(2, 2, 2)
    )
    assert c.element == "Si"
    assert c.rattle_stdev == 0.0 # Default

def test_full_config_with_new_sections() -> None:
    """Test Config with all sections."""
    c = Config(
        project_name="Full",
        potential=PotentialConfig(elements=["Si"], cutoff=4.0),
        lammps=LammpsConfig(command="lmp_mpi", cores=4),
        structure_gen=StructureGenConfig(
            element="Si",
            crystal_structure="diamond",
            lattice_constant=5.43,
            supercell=(2,2,2),
            rattle_stdev=0.1
        )
    )
    assert c.lammps.cores == 4
    assert c.structure_gen.rattle_stdev == 0.1
