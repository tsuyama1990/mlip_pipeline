from unittest.mock import MagicMock, patch

from mlip_autopipec.config.schemas.common import MinimalConfig
from mlip_autopipec.config.schemas.generator import GeneratorConfig, SQSConfig, DistortionConfig, NMSConfig, DefectConfig
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

    structures = builder.build()

    # 1 SQS (base)
    # Strains: -0.05, 0.05 (2 steps). Both non-zero. -> 2 Strained structures.
    # Total Base + Strained = 3.
    # Rattles: 3 * 2 = 6.
    # Total = 3 + 6 = 9 structures.

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

    # We can mock DefectApplicator to ensure separation
    with patch('mlip_autopipec.generator.builder.DefectApplicator') as MockApplicator:
        instance = MockApplicator.return_value
        # Mock apply to return 1 structure (dummy)
        from ase import Atoms
        instance.apply.return_value = [Atoms('Fe')]

        builder = StructureBuilder(system_config)
        structures = builder.build()

        assert len(structures) == 1
        instance.apply.assert_called_once()
