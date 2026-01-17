from unittest.mock import MagicMock

from mlip_autopipec.config.schemas.common import MinimalConfig
from mlip_autopipec.config.schemas.generator import (
    DefectConfig,
    DistortionConfig,
    GeneratorConfig,
    NMSConfig,
    SQSConfig,
)
from mlip_autopipec.config.schemas.system import SystemConfig, TargetSystem
from mlip_autopipec.generator.builder import StructureBuilder


def test_builder_pipeline():
    # Setup Config using updated schema hierarchy with explicit initialization
    gen_config = GeneratorConfig(
        sqs=SQSConfig(enabled=True, supercell_matrix=[[2,0,0], [0,2,0], [0,0,2]]),
        distortion=DistortionConfig(enabled=True, n_strain_steps=2, n_rattle_steps=2),
        nms=NMSConfig(enabled=True),
        defects=DefectConfig(enabled=False)
    )

    target_system = TargetSystem(
        name="TestSystem",
        composition={"Fe": 1.0},
        elements=["Fe"],
        structure_type="bulk"
    )

    minimal_config = MinimalConfig(
        project_name="test_proj",
        target_system=target_system,
        resources={"dft_code": "quantum_espresso", "parallel_cores": 4}
    )

    system_config = MagicMock(spec=SystemConfig)
    system_config.generator_config = gen_config
    system_config.target_system = target_system
    system_config.minimal = minimal_config

    builder = StructureBuilder(system_config)

    # In StructureBuilder.__init__, it checks for self.system_config.generator_config
    # However, since we mock SystemConfig, we must ensure the attribute access returns our gen_config.
    # MagicMock attributes work, but pydantic access pattern in builder might vary if not using dict access.
    # The builder code: self.generator_config = system_config.generator_config

    structures = builder.build()

    assert len(structures) == 9
    for s in structures:
        assert isinstance(s.info.get('uuid'), str)
        assert 'config_type' in s.info

def test_builder_with_defects():
    gen_config = GeneratorConfig(
        sqs=SQSConfig(enabled=True),
        distortion=DistortionConfig(enabled=False),
        nms=NMSConfig(enabled=True),
        defects=DefectConfig(enabled=True, vacancies=True, interstitials=False)
    )

    target_system = TargetSystem(
        name="TestSystem",
        composition={"Fe": 1.0},
        elements=["Fe"],
        structure_type="bulk"
    )

    # Mock SystemConfig
    system_config = MagicMock(spec=SystemConfig)
    system_config.generator_config = gen_config
    system_config.target_system = target_system

    builder = StructureBuilder(system_config)
    structures = builder.build()

    assert len(structures) == 9
    types = [s.info.get('config_type') for s in structures]
    assert 'sqs' in types
    assert 'vacancy' in types
