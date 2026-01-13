
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms
from ase.build import bulk

from mlip_autopipec.modules.a_generator import generate_structures
from mlip_autopipec.schemas.system_config import GeneratorParams, SystemConfig
from mlip_autopipec.schemas.user_config import (
    GenerationConfig,
    TargetSystem,
    UserConfig,
)


@pytest.fixture
def mock_user_config() -> UserConfig:
    """Returns a mock UserConfig for an FeNi alloy."""
    return UserConfig(
        project_name="TestProject",
        target_system=TargetSystem(
            elements=["Fe", "Ni"],
            composition={"Fe": 0.5, "Ni": 0.5},
            crystal_structure="fcc",
        ),
        generation_config=GenerationConfig(generation_type="alloy_sqs"),
    )


@pytest.fixture
def mock_system_config() -> SystemConfig:
    """Returns a mock SystemConfig with generator parameters."""
    return SystemConfig(
        generator_params=GeneratorParams(
            generation_type="alloy_sqs",
            sqs_supercell_size=[2, 2, 2],
            strain_magnitudes=[0.01, -0.01],
            rattle_std_dev=0.05,
        ),
        dft_params=MagicMock(),  # DFT params are not needed for this test
    )


@patch("mlip_autopipec.modules.a_generator._generate_alloy_sqs")
def test_generate_structures_alloy_sqs(
    mock_generate_sqs: MagicMock,
    mock_user_config: UserConfig,
    mock_system_config: SystemConfig,
):
    """
    Tests that the main structure generation function correctly dispatches to the SQS
    generator and applies the specified strains and rattles.
    """
    # Configure the mock to return a single, simple Atoms object
    mock_sqs_atoms = bulk("Fe", "fcc", a=3.6, cubic=True)
    mock_generate_sqs.return_value = [mock_sqs_atoms]

    generated_structures = generate_structures(mock_user_config, mock_system_config)

    # Verify that the SQS generator was called once with the correct parameters
    mock_generate_sqs.assert_called_once_with(
        mock_user_config.target_system.elements,
        mock_user_config.target_system.composition,
        mock_system_config.generator_params.sqs_supercell_size,
        mock_user_config.target_system.crystal_structure,
    )

    # Check the total number of generated structures
    # 1 SQS * 2 strain levels * 1 rattle level = 2 structures
    assert len(generated_structures) == 2

    # Check that strains and rattles were applied correctly
    for atoms in generated_structures:
        assert isinstance(atoms, Atoms)
        assert "config_type" in atoms.info
        assert "sqs_strain" in atoms.info["config_type"]
        assert "rattle" in atoms.info["config_type"]

        # Check that the cell volume was changed by the strain
        original_volume = mock_sqs_atoms.get_volume()
        assert not abs(atoms.get_volume() - original_volume) < 1e-6

        # Check that atoms were rattled (positions are not the same as original)
        assert not (atoms.positions == mock_sqs_atoms.positions).all()


def test_generate_structures_not_implemented(
    mock_user_config: UserConfig, mock_system_config: SystemConfig
):
    """
    Tests that a NotImplementedError is raised for unsupported generation types.
    """
    mock_user_config.generation_config.generation_type = "unsupported_type"
    with pytest.raises(NotImplementedError):
        generate_structures(mock_user_config, mock_system_config)
