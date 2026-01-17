from unittest.mock import MagicMock

from mlip_autopipec.config.schemas.common import MinimalConfig
from mlip_autopipec.config.schemas.generator import GeneratorConfig
from mlip_autopipec.config.schemas.system import SystemConfig, TargetSystem
from mlip_autopipec.generator.builder import StructureBuilder


def test_builder_pipeline():
    # Setup Config using updated schema hierarchy
    gen_config = GeneratorConfig()
    gen_config.sqs.supercell_matrix = [[2,0,0], [0,2,0], [0,0,2]]
    gen_config.distortion.n_strain_steps = 2
    gen_config.distortion.n_rattle_steps = 2

    # Disable defects for predictable count in basic test
    gen_config.defects.enabled = False

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

    structures = builder.build()

    # 1 SQS (base)
    # Strains: -0.05, 0.05 (2 steps). Both non-zero. -> 2 Strained structures.
    # Total Base + Strained = 3.
    # Rattles: 3 * 2 = 6.
    # Total = 3 + 6 = 9 structures.

    # Wait, in AlloyGenerator.generate_batch:
    # 2 strains: linspace(-0.05, 0.05, 2) -> [-0.05, 0.05]. Correct.

    assert len(structures) == 9
    for s in structures:
        assert isinstance(s.info.get('uuid'), str)
        assert 'config_type' in s.info

def test_builder_with_defects():
    gen_config = GeneratorConfig()
    gen_config.distortion.enabled = False # Simplify
    gen_config.defects.enabled = True
    gen_config.defects.vacancies = True

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

    # 1 SQS (Base)
    # Vacancies: 8 atoms (2x2x2) -> 8 vacancies.
    # Total = 9.

    assert len(structures) == 9
    types = [s.info.get('config_type') for s in structures]
    assert 'sqs' in types
    assert 'vacancy' in types
