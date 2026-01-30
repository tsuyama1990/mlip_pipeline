from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import (
    Config,
    MDConfig,
    PotentialConfig,
    StructureGenConfig,
)


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
            strategy="bulk",
            element="Ti",
            crystal_structure="hcp",
            lattice_constant=2.95,
        ),
        md=MDConfig(
            temperature=300.0,
            n_steps=100,
            timestep=0.001,
            ensemble="NVT"
        )
    )
    assert c.project_name == "TestProject"
    assert c.potential.cutoff == 5.0
    assert c.logging.level == "INFO" # Default
    assert c.structure_gen.element == "Ti"
    assert c.md.temperature == 300.0


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
                element="Ti", crystal_structure="hcp", lattice_constant=2.95
            ),
            md=MDConfig(
                temperature=300, n_steps=100, ensemble="NVT"
            )
        )
    assert "Cutoff must be greater than 0" in str(excinfo.value)


def test_config_missing_field() -> None:
    """Test missing required field."""
    with pytest.raises(ValidationError):
        Config(
            project_name="Test",
            potential=None, # type: ignore[arg-type]
            # Missing potential, etc.
        ) # type: ignore[call-arg]


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
      strategy: "bulk"
      element: "Cu"
      crystal_structure: "fcc"
      lattice_constant: 3.61
    md:
      temperature: 500.0
      n_steps: 500
      timestep: 0.002
      ensemble: "NVT"
    """
    yaml_file.write_text(yaml_content)

    c = Config.from_yaml(yaml_file)

    assert c.project_name == "YamlProject"
    assert c.potential.elements == ["Cu"]
    assert c.logging.level == "DEBUG"
    assert c.structure_gen.crystal_structure == "fcc"
    assert c.md.n_steps == 500
