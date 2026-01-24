
import pytest

from mlip_autopipec.config.models import SystemConfig
from mlip_autopipec.config.schemas.core import TargetSystem
from mlip_autopipec.config.schemas.generator import GeneratorConfig
from mlip_autopipec.generator.builder import StructureBuilder


@pytest.fixture
def system_config() -> SystemConfig:
    target = TargetSystem(
        name="AlCu",
        elements=["Al", "Cu"],
        composition={"Al": 0.5, "Cu": 0.5},
        crystal_structure="fcc"
    )
    gen_config = GeneratorConfig(number_of_structures=5, seed=42)
    return SystemConfig(target_system=target, generator_config=gen_config)

def test_builder_initialization(system_config: SystemConfig) -> None:
    builder = StructureBuilder(system_config)
    assert builder.generator_config.number_of_structures == 5

def test_build_flow(system_config: SystemConfig) -> None:
    # Disable heavy distortions/defects for fast test
    system_config.generator_config.distortion.n_strain_steps = 2
    system_config.generator_config.distortion.n_rattle_steps = 1

    builder = StructureBuilder(system_config)
    structures = builder.build()

    assert isinstance(structures, list)
    assert len(structures) > 0
    # Should be limited to number_of_structures if we generated more
    assert len(structures) <= 5

    for s in structures:
        assert "uuid" in s.info
        assert s.info["target_system"] == "AlCu"

def test_build_with_defaults(system_config: SystemConfig) -> None:
    # Default generator config
    system_config.generator_config = GeneratorConfig() # Defaults
    builder = StructureBuilder(system_config)
    structures = builder.build()
    assert len(structures) > 0

def test_build_no_target() -> None:
    config = SystemConfig() # No target system
    builder = StructureBuilder(config)
    structures = builder.build()
    assert structures == []
