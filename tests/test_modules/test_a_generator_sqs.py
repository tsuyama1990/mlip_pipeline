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


def test_generate_alloy_sqs_varied_inputs():
    """
    Test that the SQS generator works with varied inputs.
    """
    user_config = UserConfig(
        project_name="test_alloy",
        target_system=TargetSystem(
            elements=["Au", "Cu"],
            composition={"Au": 0.25, "Cu": 0.75},
            crystal_structure="fcc",
        ),
        generation_config=GenerationConfig(generation_type="alloy_sqs"),
        surrogate_config={
            "model_path": "path/to/model",
            "num_to_select_fps": 10,
            "descriptor_type": "SOAP",
        },
        trainer_config={
            "radial_basis": "bessel",
            "max_body_order": 2,
            "loss_weights": {"energy": 1.0, "forces": 100.0, "stress": 0.0},
        },
    )
    generator_params = GeneratorParams(
        sqs_supercell_size=[2, 2, 2],
        strain_magnitudes=[0],
        rattle_standard_deviation=0,
        lattice_constant=4.0,
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
        surrogate_config=user_config.surrogate_config,
        trainer_config=user_config.trainer_config,
    )

    with patch("mlip_autopipec.modules.a_generator.generate_sqs") as mock_generate_sqs:
        mock_generate_sqs.return_value = Atoms("Au", cell=[4, 4, 4], pbc=True)
        structures = generate_structures(system_config)
        assert len(structures) == 1
        mock_generate_sqs.assert_called_once()
