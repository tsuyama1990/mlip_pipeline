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
    structures = generator._generate_for_alloy()
    assert len(structures) == 12


@patch("mlip_autopipec.modules.generator.PhysicsInformedGenerator._create_vacancy")
def test_generate_crystal_workflow(
    mock_create_vacancy: MagicMock,
    base_system_config: SystemConfig,
) -> None:
    """Test the full crystal defect generation workflow with mocking."""
    mock_vacancy = MagicMock()
    mock_vacancy.lattice.matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    mock_vacancy.lattice.pbc = (True, True, True)
    mock_vacancy.species = ["H"]
    mock_vacancy.frac_coords = [[0, 0, 0]]
    mock_create_vacancy.return_value = [mock_vacancy]
    generator = PhysicsInformedGenerator(config=base_system_config)
    structures = generator._generate_for_crystal()
    assert len(structures) == 1
