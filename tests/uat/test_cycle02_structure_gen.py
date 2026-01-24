import numpy as np

from mlip_autopipec.config.models import SystemConfig
from mlip_autopipec.config.schemas.core import TargetSystem
from mlip_autopipec.config.schemas.generator import DefectConfig, DistortionConfig, GeneratorConfig
from mlip_autopipec.generator.builder import StructureBuilder


def test_scenario_02_01_bulk_generation() -> None:
    """
    Scenario 02-01: Bulk Structure Generation
    Generate simple bulk supercells with thermal noise.
    """
    target = TargetSystem(
        name="Al",
        elements=["Al"],
        composition={"Al": 1.0},
        crystal_structure="fcc"
    )

    # Enable SQS to get supercell, even for pure element
    gen_config = GeneratorConfig(
        sqs={"enabled": True, "supercell_size": [2, 2, 2]},
        distortion=DistortionConfig(
            enabled=True,
            rattle_stdev=0.1,
            strain_range=(0.0, 0.0), # Disable strain to focus on rattle
            n_strain_steps=1,
            n_rattle_steps=5
        ),
        defects=DefectConfig(enabled=False),
        number_of_structures=5,
        seed=123
    )

    sys_config = SystemConfig(target_system=target, generator_config=gen_config)
    builder = StructureBuilder(sys_config)

    structures = list(builder.build())

    # We requested 5 structures.
    assert len(structures) == 5

    coords0 = structures[0].positions
    coords1 = structures[1].positions
    assert not np.allclose(coords0, coords1)


def test_scenario_02_02_defect_introduction() -> None:
    """
    Scenario 02-02: Defect Introduction
    Create a vacancy in a supercell.
    """
    target = TargetSystem(
        name="Al",
        elements=["Al"],
        composition={"Al": 1.0},
        crystal_structure="fcc"
    )

    # Enable SQS to get supercell of 8 atoms (2x2x2 primitive)
    gen_config = GeneratorConfig(
        sqs={"enabled": True, "supercell_size": [2, 2, 2]},
        distortion={"enabled": False},
        defects=DefectConfig(enabled=True, vacancies=True, interstitials=False),
        number_of_structures=10,
        seed=123
    )

    sys_config = SystemConfig(target_system=target, generator_config=gen_config)
    builder = StructureBuilder(sys_config)

    # 2x2x2 FCC Primitive = 1 * 8 = 8 atoms

    structures = list(builder.build())

    # Check for vacancy structures
    vacancy_structs = [s for s in structures if s.info.get("config_type") == "vacancy"]
    assert len(vacancy_structs) > 0

    for s in vacancy_structs:
        assert len(s) == 7 # 8 - 1


def test_scenario_02_03_reproducibility() -> None:
    """
    Scenario 02-03: Reproducibility
    Ensure identical seeds produce identical structures.
    """
    target = TargetSystem(
        name="Al",
        elements=["Al"],
        composition={"Al": 1.0},
        crystal_structure="fcc"
    )

    gen_config = GeneratorConfig(
        sqs={"enabled": True, "supercell_size": [2, 2, 2]},
        distortion=DistortionConfig(enabled=True, rattle_stdev=0.05),
        number_of_structures=5,
        seed=42
    )

    sys_config = SystemConfig(target_system=target, generator_config=gen_config)

    # Run 1
    builder1 = StructureBuilder(sys_config)
    batch1 = list(builder1.build())

    # Run 2
    builder2 = StructureBuilder(sys_config)
    batch2 = list(builder2.build())

    assert len(batch1) == len(batch2)

    for s1, s2 in zip(batch1, batch2, strict=True):
        np.testing.assert_allclose(s1.positions, s2.positions)
        np.testing.assert_allclose(s1.cell, s2.cell)
        assert s1.info["uuid"] != s2.info["uuid"] # UUIDs should be different
