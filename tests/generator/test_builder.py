from unittest.mock import MagicMock

from mlip_autopipec.config.schemas.common import MinimalConfig
from mlip_autopipec.config.schemas.generator import GeneratorConfig
from mlip_autopipec.config.schemas.system import SystemConfig, TargetSystem
from mlip_autopipec.generator.builder import StructureBuilder


def test_builder_pipeline():
    # Setup Config
    gen_config = GeneratorConfig(
        supercell_matrix=[[2,0,0], [0,2,0], [0,0,2]],
        n_strain_steps=2,
        n_rattle_steps=2
    )

    # Note: TargetSystem has 'elements' field required.
    target_system = TargetSystem(
        name="TestSystem",
        composition={"Fe": 1.0}, # Pure element for simplicity
        elements=["Fe"], # Must match keys of composition
        structure_type="bulk"
    )

    minimal_config = MinimalConfig(
        project_name="test_proj",
        target_system=target_system,
        # MinimalConfig also requires 'resources'
        resources={"dft_code": "quantum_espresso", "parallel_cores": 4}
    )

    system_config = MagicMock(spec=SystemConfig)
    system_config.generator_config = gen_config
    system_config.target_system = target_system
    system_config.minimal = minimal_config

    builder = StructureBuilder(system_config)

    structures = builder.build()

    # 1 SQS (base) -> 1
    # 2 strain steps -> check if they are non-zero.
    # default range (-0.05, 0.05). Linspace(2) -> -0.05, 0.05. Both non-zero.
    # So 1 Base + 2 Strains = 3 structures.
    # Rattles: each of 3 gets 2 rattles -> 6 rattles.
    # Total = 3 + 6 = 9 structures.
    # Plus Vacancies? default SQS size 8 atoms.
    # DefectGenerator.create_vacancy(sqs) -> 8 vacancies (naive).
    # Total = 9 + 8 = 17 structures.

    assert len(structures) > 0
    for s in structures:
        assert isinstance(s.info.get('uuid'), str)
        assert 'config_type' in s.info
