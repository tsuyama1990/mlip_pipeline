from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import Config, LammpsConfig, PotentialConfig


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
    # Check new defaults
    assert c.structure_gen.lattice_constant == 5.43
    assert c.structure_gen.md_params.temperature == 300.0

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
      element_params:
        Cu:
            mass: 63.546
            lj_sigma: 2.338
            lj_epsilon: 0.167
            zbl_z: 29
    logging:
      level: "DEBUG"
    lammps:
      command: "mpirun -np 4 lmp_mpi"
      timeout: 600
      cores: 4
    structure_gen:
      md_params:
        temperature: 500
        n_steps: 2000
      lattice_constant: 3.61
    """
    yaml_file.write_text(yaml_content)

    c = Config.from_yaml(yaml_file)

    assert c.project_name == "YamlProject"
    assert c.potential.elements == ["Cu"]
    assert c.potential.element_params["Cu"].zbl_z == 29
    assert c.logging.level == "DEBUG"
    assert c.lammps.cores == 4
    assert c.lammps.timeout == 600
    assert c.structure_gen.md_params.temperature == 500
    assert c.structure_gen.lattice_constant == 3.61

def test_lammps_config_extra_forbid() -> None:
    """Test that extra fields are forbidden in LammpsConfig."""
    with pytest.raises(ValidationError):
        LammpsConfig(extra_field="fail") # type: ignore
