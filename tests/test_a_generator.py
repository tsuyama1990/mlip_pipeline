import pytest

from mlip_autopipec.modules.a_generator import generate_structures
from mlip_autopipec.schemas.system_config import DFTParams, GeneratorParams, SystemConfig
from mlip_autopipec.schemas.user_config import UserConfig


def test_generate_structures_invalid_generation_type():
    """Test that generate_structures raises ValueError for invalid generation_type."""
    user_config_data = {
        "project_name": "test_project",
        "target_system": {
            "elements": ["Fe", "Ni"],
            "composition": {"Fe": 0.5, "Ni": 0.5},
            "crystal_structure": "fcc",
        },
        "generation_config": {"generation_type": "invalid_type"},
        "surrogate_config": {
            "calculator": "mace_mp",
            "model_path": "path/to/model",
            "num_to_select_fps": 10,
            "descriptor_type": "SOAP",
        },
        "trainer_config": {
            "radial_basis": "bessel",
            "max_body_order": 2,
            "loss_weights": {"energy": 1.0, "forces": 100.0, "stress": 0.0},
        },
    }
    user_config = UserConfig.model_validate(user_config_data)

    system_config = SystemConfig(
        user_config=user_config,
        dft_params=DFTParams(),
        generator_params=GeneratorParams(
            sqs_supercell_size=[2, 2, 2],
            strain_magnitudes=[-0.05, 0.05],
            rattle_standard_deviation=0.1,
        ),
        surrogate_config=user_config.surrogate_config,
        trainer_config=user_config.trainer_config,
    )

    with pytest.raises(ValueError, match="Unknown generation type: invalid_type"):
        generate_structures(system_config)


def test_generate_structures_invalid_supercell_size():
    """Test that generate_structures raises ValueError for invalid sqs_supercell_size."""
    user_config_data = {
        "project_name": "test_project",
        "target_system": {
            "elements": ["Fe", "Ni"],
            "composition": {"Fe": 0.5, "Ni": 0.5},
            "crystal_structure": "fcc",
        },
        "generation_config": {"generation_type": "alloy_sqs"},
        "surrogate_config": {
            "calculator": "mace_mp",
            "model_path": "path/to/model",
            "num_to_select_fps": 10,
            "descriptor_type": "SOAP",
        },
        "trainer_config": {
            "radial_basis": "bessel",
            "max_body_order": 2,
            "loss_weights": {"energy": 1.0, "forces": 100.0, "stress": 0.0},
        },
    }
    user_config = UserConfig.model_validate(user_config_data)

    system_config = SystemConfig(
        user_config=user_config,
        dft_params=DFTParams(),
        generator_params=GeneratorParams(
            sqs_supercell_size=[2, 2],  # Invalid size
            strain_magnitudes=[-0.05, 0.05],
            rattle_standard_deviation=0.1,
        ),
        surrogate_config=user_config.surrogate_config,
        trainer_config=user_config.trainer_config,
    )

    with pytest.raises(ValueError, match="sqs_supercell_size must be a list of three integers."):
        generate_structures(system_config)
