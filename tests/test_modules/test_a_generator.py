from unittest.mock import patch

from ase.atoms import Atoms

from mlip_autopipec.modules.a_generator import generate_structures
from mlip_autopipec.schemas.system_config import (
    DFTParams,
    GeneratorParams,
    SystemConfig,
)
from mlip_autopipec.schemas.user_config import (
    GenerationConfig,
    TargetSystem,
    UserConfig,
)


def test_generate_alloy_sqs_structures() -> None:
    """
    Test that the correct number of SQS structures are generated for an alloy.
    """
    user_config = UserConfig(
        project_name="test_alloy",
        target_system=TargetSystem(
            elements=["Fe", "Ni"],
            composition={"Fe": 0.5, "Ni": 0.5},
            crystal_structure="fcc",
        ),
        generation_config=GenerationConfig(generation_type="alloy_sqs"),
    )
    generator_params = GeneratorParams(
        sqs_supercell_size=[2, 2, 2],
        strain_magnitudes=[-0.01, 0, 0.01],  # 3 strain values
        rattle_standard_deviation=0.05,
    )
    system_config = SystemConfig(
        user_config=user_config,
        dft_params=DFTParams(  # Dummy params
            pseudopotentials={},
            cutoff_wfc=0,
            k_points=(0, 0, 0),
            smearing="",
            degauss=0,
            nspin=1,
        ),
        generator_params=generator_params,
    )

    # Mock the external `generate_sqs` function where it is used in the a_generator module
    with patch("mlip_autopipec.modules.a_generator.generate_sqs") as mock_generate_sqs:
        # Configure the mock to return a simple, single-atom structure
        mock_generate_sqs.return_value = Atoms("Fe", cell=[1, 1, 1], pbc=True)

        structures = generate_structures(system_config)

        # We expect one structure for each strain value
        assert len(structures) == len(generator_params.strain_magnitudes)
        mock_generate_sqs.assert_called_once()


def test_generate_eos_strain_structures() -> None:
    """
    Test that the correct number of EOS strain structures are generated.
    """
    user_config = UserConfig(
        project_name="test_eos",
        target_system=TargetSystem(
            elements=["Si"],
            composition={"Si": 1.0},
            crystal_structure="diamond",
        ),
        generation_config=GenerationConfig(generation_type="eos_strain"),
    )
    generator_params = GeneratorParams(
        sqs_supercell_size=[],
        strain_magnitudes=[-0.05, -0.02, 0, 0.02, 0.05],  # 5 strain values
        rattle_standard_deviation=0,
    )
    system_config = SystemConfig(
        user_config=user_config,
        dft_params=DFTParams(
            pseudopotentials={},
            cutoff_wfc=0,
            k_points=(0, 0, 0),
            smearing="",
            degauss=0,
            nspin=1,
        ),
        generator_params=generator_params,
    )

    structures = generate_structures(system_config)
    assert len(structures) == len(generator_params.strain_magnitudes)


def test_generate_nms_structures() -> None:
    """Test NMS structure generation."""
    user_config = UserConfig(
        project_name="test_nms",
        target_system=TargetSystem(
            elements=["H"],
            composition={"H": 1.0},
            crystal_structure="sc",
        ),
        generation_config=GenerationConfig(generation_type="nms"),
    )
    generator_params = GeneratorParams(
        sqs_supercell_size=[],
        strain_magnitudes=[],
        rattle_standard_deviation=0,
    )
    system_config = SystemConfig(
        user_config=user_config,
        dft_params=DFTParams(
            pseudopotentials={},
            cutoff_wfc=0,
            k_points=(0, 0, 0),
            smearing="",
            degauss=0,
            nspin=1,
        ),
        generator_params=generator_params,
    )

    structures = generate_structures(system_config)
    assert len(structures) == 6


def test_generate_melt_quench_structures() -> None:
    """Test melt-quench structure generation."""
    user_config = UserConfig(
        project_name="test_melt_quench",
        target_system=TargetSystem(
            elements=["Cu"],
            composition={"Cu": 1.0},
            crystal_structure="fcc",
        ),
        generation_config=GenerationConfig(generation_type="melt_quench"),
    )
    generator_params = GeneratorParams(
        sqs_supercell_size=[],
        strain_magnitudes=[],
        rattle_standard_deviation=0,
    )
    system_config = SystemConfig(
        user_config=user_config,
        dft_params=DFTParams(
            pseudopotentials={},
            cutoff_wfc=0,
            k_points=(0, 0, 0),
            smearing="",
            degauss=0,
            nspin=1,
        ),
        generator_params=generator_params,
    )

    structures = generate_structures(system_config)
    assert len(structures) == 1
