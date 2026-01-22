from ase.build import bulk

from mlip_autopipec.config.models import SystemConfig
from mlip_autopipec.config.schemas.common import TargetSystem
from mlip_autopipec.config.schemas.generator import DistortionConfig, GeneratorConfig, SQSConfig
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
    # We mock SQSStrategy.generate to return something specific
    mocker.patch("mlip_autopipec.generator.builder.SQSStrategy.generate", return_value=bulk("Fe"))
    # We mock apply_strain and apply_rattle in builder module scope
    mocker.patch("mlip_autopipec.generator.builder.apply_strain", side_effect=lambda a, s: a)
    mocker.patch("mlip_autopipec.generator.builder.apply_rattle", side_effect=lambda a, s: a)

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
