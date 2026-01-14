# ruff: noqa: D100, D103, S101
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.config_schemas import SystemConfig, TargetSystem
from mlip_autopipec.modules.generator import PhysicsInformedGenerator


@pytest.fixture
def base_system_config() -> SystemConfig:
    """Provide a base SystemConfig for testing the generator."""
    target_system = TargetSystem(
        elements=["H", "He"], composition={"H": 0.5, "He": 0.5}
    )
    # This is a placeholder and will need to be filled in with a valid config
    return SystemConfig(
        target_system=target_system,
        dft={"executable": {}, "input": {"pseudopotentials": {}}},
    )


@pytest.fixture
def crystal_system_config() -> SystemConfig:
    """Provide a SystemConfig for testing the crystal generator."""
    target_system = TargetSystem(elements=["H"], composition={"H": 1.0})
    # This is a placeholder and will need to be filled in with a valid config
    return SystemConfig(
        target_system=target_system,
        dft={"executable": {}, "input": {"pseudopotentials": {}}},
    )


def test_apply_strains(base_system_config: SystemConfig) -> None:
    """Test the application of volumetric and shear strains."""
    atoms = Atoms("H", cell=[1, 1, 1], pbc=True)
    generator = PhysicsInformedGenerator(config=base_system_config)
    strained_atoms = generator._apply_strains(atoms)
    assert len(strained_atoms) == 3
    assert strained_atoms[0].cell[0][0] == 0.95


def test_apply_rattling(base_system_config: SystemConfig) -> None:
    """Test the application of atomic rattling."""
    atoms = Atoms("H", cell=[1, 1, 1], pbc=True)
    generator = PhysicsInformedGenerator(config=base_system_config)
    rattled_atoms = generator._apply_rattling(atoms)
    assert len(rattled_atoms) == 2
    assert not np.allclose(rattled_atoms[0].positions, atoms.positions)


@patch("mlip_autopipec.modules.generator.PhysicsInformedGenerator._create_sqs_structure")
def test_generate_alloy_workflow(
    mock_create_sqs: MagicMock, base_system_config: SystemConfig
) -> None:
    """Test the full alloy generation workflow with mocking."""
    mock_sqs_atoms = Atoms(
        "H2", positions=[[0, 0, 0], [0, 0, 1]], cell=[2, 2, 2], pbc=True
    )
    mock_create_sqs.return_value = mock_sqs_atoms
    generator = PhysicsInformedGenerator(config=base_system_config)
    structures = generator.generate()
    assert len(structures) == 12
    assert isinstance(structures[0].atoms, Atoms)


@patch("mlip_autopipec.modules.generator.PhysicsInformedGenerator._create_vacancy")
def test_generate_crystal_workflow(
    mock_create_vacancy: MagicMock,
    crystal_system_config: SystemConfig,
) -> None:
    """Test the full crystal defect generation workflow with mocking."""
    mock_vacancy = MagicMock()
    mock_vacancy.lattice.matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    mock_vacancy.lattice.pbc = (True, True, True)
    mock_vacancy.species = ["H"]
    mock_vacancy.frac_coords = [[0, 0, 0]]
    mock_create_vacancy.return_value = [mock_vacancy]
    generator = PhysicsInformedGenerator(config=crystal_system_config)
    structures = generator.generate()
    assert len(structures) == 1
    assert isinstance(structures[0].atoms, Atoms)


@patch("mlip_autopipec.modules.generator.PhysicsInformedGenerator._generate_for_alloy")
@patch("mlip_autopipec.modules.generator.PhysicsInformedGenerator._generate_for_crystal")
def test_generate_dispatch(
    mock_generate_crystal: MagicMock,
    mock_generate_alloy: MagicMock,
    base_system_config: SystemConfig,
) -> None:
    """Test that the generate method dispatches to the correct workflow."""
    # Test alloy case
    alloy_config = base_system_config.model_copy(deep=True)
    alloy_config.target_system.elements = ["H", "He"]
    generator = PhysicsInformedGenerator(config=alloy_config)
    generator.generate()
    mock_generate_alloy.assert_called_once()
    mock_generate_crystal.assert_not_called()

    # Test crystal case
    mock_generate_alloy.reset_mock()
    crystal_config = base_system_config.model_copy(deep=True)
    crystal_config.target_system.elements = ["H"]
    generator = PhysicsInformedGenerator(config=crystal_config)
    generator.generate()
    mock_generate_alloy.assert_not_called()
    mock_generate_crystal.assert_called_once()
