import pytest
import numpy as np
from ase import Atoms
from ase.build import bulk

from mlip_autopipec.config.models import SystemConfig
from mlip_autopipec.config.schemas.common import TargetSystem
from mlip_autopipec.config.schemas.generator import GeneratorConfig, SQSConfig, DistortionConfig
from mlip_autopipec.generator.builder import StructureBuilder


def test_builder_initialization():
    gen_config = GeneratorConfig()
    sys_config = SystemConfig(
        target_system=TargetSystem(
            name="Test", elements=["Fe"], composition={"Fe": 1.0}
        ),
        generator_config=gen_config
    )
    builder = StructureBuilder(sys_config)
    assert builder.generator_config == gen_config


def test_builder_build_batch_simple(mocker):
    # Mock strategies to avoid heavy lifting
    mocker.patch("mlip_autopipec.generator.builder.SQSStrategy.generate", return_value=bulk("Fe"))
    # We mock apply_strain and apply_rattle in builder module scope
    mocker.patch("mlip_autopipec.generator.builder.apply_strain", side_effect=lambda a, s: a)
    mocker.patch("mlip_autopipec.generator.builder.apply_rattle", side_effect=lambda a, s, rng=None: a)

    gen_config = GeneratorConfig(
        sqs=SQSConfig(enabled=True),
        distortion=DistortionConfig(enabled=True, n_strain_steps=2, n_rattle_steps=1),
        number_of_structures=5
    )
    sys_config = SystemConfig(
        target_system=TargetSystem(
            name="FeNi",
            elements=["Fe", "Ni"],
            composition={"Fe": 0.5, "Ni": 0.5}
        ),
        generator_config=gen_config
    )

    builder = StructureBuilder(sys_config)
    results = builder.build_batch()

    assert len(results) == 5
    for atoms in results:
        assert "uuid" in atoms.info
        assert atoms.info["target_system"] == "FeNi"

def test_builder_content_verification():
    """Verify that the builder produces atoms with expected properties (volume change for strain)."""
    gen_config = GeneratorConfig(
        sqs=SQSConfig(enabled=False),
        distortion=DistortionConfig(
            enabled=True,
            strain_range=(-0.1, 0.1), # significant strain
            n_strain_steps=3, # -0.1, 0.0, 0.1
            n_rattle_steps=0
        ),
        defects={"enabled": False},
        number_of_structures=10,
        seed=42
    )
    sys_config = SystemConfig(
        target_system=TargetSystem(
            name="Cu", elements=["Cu"], composition={"Cu": 1.0}, crystal_structure="fcc"
        ),
        generator_config=gen_config
    )

    builder = StructureBuilder(sys_config)
    results = builder.build_batch()

    # Expect 3 structures: Base (0.0), Strained (-0.1), Strained (0.1)
    # n_strain_steps=3 on [-0.1, 0.1] -> [-0.1, 0.0, 0.1]. 0.0 is skipped in loop if logic holds.
    # Logic: if abs(s) < 1e-6: continue.
    # Base structure is added at start.
    # So we expect 1 Base + 2 Strained = 3.
    assert len(results) == 3

    base_vol = results[0].get_volume()
    # Check volumes
    volumes = [a.get_volume() for a in results]
    # Sort volumes
    volumes.sort()

    # Smallest (0.9^3 ~ 0.729)
    assert np.isclose(volumes[0], base_vol * 0.9**3)
    # Largest (1.1^3 ~ 1.331)
    assert np.isclose(volumes[-1], base_vol * 1.1**3)

def test_builder_determinism():
    """Verify that seeding produces identical results."""
    sys_config = SystemConfig(
        target_system=TargetSystem(
            name="Cu", elements=["Cu"], composition={"Cu": 1.0}, crystal_structure="fcc"
        ),
        generator_config=GeneratorConfig(
            distortion={"enabled": True, "rattle_stdev": 0.1, "n_rattle_steps": 1},
            seed=123
        )
    )

    b1 = StructureBuilder(sys_config)
    res1 = b1.build_batch()

    b2 = StructureBuilder(sys_config)
    res2 = b2.build_batch()

    # Check positions of rattled structures
    # Assuming index 1 is a rattled structure (Index 0 is base, strain disabled by default range but enabled generally)
    # Wait, distortion defaults: strain_range (-0.05, 0.05), steps 5.
    # So we have many structures.
    # Check the last one.

    atoms1 = res1[-1]
    atoms2 = res2[-1]

    assert np.allclose(atoms1.get_positions(), atoms2.get_positions())

    # Different seed
    sys_config.generator_config.seed = 999
    b3 = StructureBuilder(sys_config)
    res3 = b3.build_batch()

    atoms3 = res3[-1]
    assert not np.allclose(atoms1.get_positions(), atoms3.get_positions())
