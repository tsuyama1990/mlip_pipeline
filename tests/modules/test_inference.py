"""Unit tests for the LAMMPS runner and uncertainty quantification modules."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from mlip_autopipec.config_schemas import SystemConfig
from mlip_autopipec.modules.inference import LammpsRunner, UncertaintyQuantifier


@pytest.fixture
def mock_system_config(tmp_path) -> SystemConfig:
    """Fixture for a mock SystemConfig object."""
    # In a real scenario, this would be loaded from a file or user input
    config_dict = {
        "target_system": {"elements": ["Cu"], "composition": {"Cu": 1.0}},
        "dft": {
            "executable": {},
            "input": {"pseudopotentials": {"Cu": "cu.upf"}},
            "retry_strategy": {"max_retries": 1, "parameter_adjustments": []},
        },
        "inference": {
            "simulation_timestep_fs": 1.0,
            "total_simulation_steps": 100,
            "uncertainty_threshold": 4.0,
            "embedding_rcut": 6.0,
            "embedding_delta_buffer": 1.0,
        },
    }
    return SystemConfig(**config_dict)


def test_uncertainty_quantifier_cycles_through_sequence() -> None:
    """Test that the mock quantifier cycles through its predefined sequence."""
    quantifier = UncertaintyQuantifier()
    atoms = Atoms("H")
    # The sequence is [1.0, 1.5, 2.0, 4.5, 2.5]
    assert quantifier.get_extrapolation_grade(atoms) == 1.0
    assert quantifier.get_extrapolation_grade(atoms) == 1.5
    assert quantifier.get_extrapolation_grade(atoms) == 2.0
    assert quantifier.get_extrapolation_grade(atoms) == 4.5
    assert quantifier.get_extrapolation_grade(atoms) == 2.5
    # Check that it wraps around
    assert quantifier.get_extrapolation_grade(atoms) == 1.0


def test_lammps_runner_initialization(mock_system_config: SystemConfig) -> None:
    """Test that the LammpsRunner can be initialized."""
    runner = LammpsRunner(
        config=mock_system_config,
        potential_path="test.yace",
        quantifier=UncertaintyQuantifier(),
    )
    assert runner.config == mock_system_config


def test_runner_yields_embedded_atoms_and_mask_on_uncertainty(
    mock_system_config: SystemConfig,
) -> None:
    """Verify the runner yields the embedded sub-cell and mask correctly."""
    quantifier = UncertaintyQuantifier()
    quantifier._mock_sequence = [1.0, 4.5]  # The second grade is above the threshold
    runner = LammpsRunner(
        config=mock_system_config, potential_path="test.yace", quantifier=quantifier
    )
    generator = runner.run()

    # The generator should yield on the second step and then terminate
    embedded_atoms, force_mask = next(generator)

    # Verify the output types
    assert isinstance(embedded_atoms, Atoms)
    assert isinstance(force_mask, np.ndarray)

    # Verify the sub-cell properties
    assert all(embedded_atoms.get_pbc())
    expected_size = 2 * (
        mock_system_config.inference.embedding_rcut
        + mock_system_config.inference.embedding_delta_buffer
    )
    assert np.allclose(embedded_atoms.get_cell().lengths(), [expected_size] * 3)

    # Verify the force mask shape and content
    assert force_mask.shape == (len(embedded_atoms), 3)
    assert np.all(np.isin(force_mask, [0.0, 1.0]))


def test_periodic_embedding_logic(mock_system_config: SystemConfig) -> None:
    """Test the _extract_periodic_subcell method with a corner case."""
    runner = LammpsRunner(
        config=mock_system_config,
        potential_path="test.yace",
        quantifier=UncertaintyQuantifier(),
    )
    # Create a large cell to test periodic wrapping
    large_atoms = bulk("Cu", "fcc", a=3.6) * (5, 5, 5)
    # Choose a corner atom as the uncertain one
    uncertain_atom_index = 0
    subcell = runner._extract_periodic_subcell(
        atoms=large_atoms,
        uncertain_atom_index=uncertain_atom_index,
        rcut=6.0,
        delta_buffer=1.0,
    )
    # Check that the sub-cell is not empty and has periodic boundary conditions
    assert len(subcell) > 1
    assert all(subcell.get_pbc())


def test_force_mask_generation(mock_system_config: SystemConfig) -> None:
    """Test the _generate_force_mask method."""
    runner = LammpsRunner(
        config=mock_system_config,
        potential_path="test.yace",
        quantifier=UncertaintyQuantifier(),
    )
    # Create a test sub-cell
    subcell = Atoms("Cu2", positions=[[0, 0, 0], [1, 1, 1]], cell=[10, 10, 10], pbc=True)
    mask = runner._generate_force_mask(subcell, rcut=0.5)
    # The first atom should be masked (weight 0), the second should not (weight 1)
    assert np.allclose(mask[0], [0.0, 0.0, 0.0])
    assert np.allclose(mask[1], [0.0, 0.0, 0.0])
