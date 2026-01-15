from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from mlip_autopipec.config_schemas import SystemConfig
from mlip_autopipec.modules.inference import LammpsRunner, UncertaintyQuantifier


@pytest.fixture
def mock_system_config() -> SystemConfig:
    """Provide a mock SystemConfig for testing the LammpsRunner."""
    config_dict = {
        "target_system": {"elements": ["Cu"], "composition": {"Cu": 1.0}},
        "dft": {
            "executable": {"command": "pw.x"},
            "input": {"pseudopotentials": {"Cu": "Cu.UPF"}},
        },
        "inference": {
            "uncertainty_threshold": 4.0,
            "total_simulation_steps": 10,
        },
    }
    return SystemConfig(**config_dict)


def test_lammps_runner_initialization(mock_system_config: SystemConfig) -> None:
    """Test that the LammpsRunner initializes correctly."""
    quantifier = UncertaintyQuantifier()
    runner = LammpsRunner(
        config=mock_system_config, potential_path="test.yace", quantifier=quantifier
    )
    assert runner.config == mock_system_config
    assert runner.potential_path == "test.yace"
    assert runner.quantifier == quantifier


def test_runner_raises_error_if_inference_config_missing(
    mock_system_config: SystemConfig,
) -> None:
    """Test that ValueError is raised if inference config is missing."""
    mock_system_config.inference = None
    with pytest.raises(ValueError, match="Inference parameters must be defined in the config."):
        LammpsRunner(
            config=mock_system_config,
            potential_path="test.yace",
            quantifier=UncertaintyQuantifier(),
        )


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
    assert isinstance(embedded_atoms, Atoms)
    assert isinstance(force_mask, np.ndarray)
    assert embedded_atoms.get_pbc().all()  # type: ignore[no-untyped-call]
    assert len(embedded_atoms) < 32  # Original structure was 32 atoms

    # Verify the generator is exhausted
    with pytest.raises(StopIteration):
        next(generator)


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
    # Assert that the number of atoms is reasonable for the given cutoff
    assert len(subcell) > 50
    assert subcell.get_pbc().all()  # type: ignore[no-untyped-call]


def test_force_mask_generation(mock_system_config: SystemConfig) -> None:
    """Test the _generate_force_mask method."""
    runner = LammpsRunner(
        config=mock_system_config,
        potential_path="test.yace",
        quantifier=UncertaintyQuantifier(),
    )
    atoms = Atoms("Cu", positions=[(5, 5, 5)], cell=[10, 10, 10], pbc=True)
    mask = runner._generate_force_mask(subcell_atoms=atoms, rcut=1.0)
    # The atom is at the center, so the distance is 0, which is less than rcut.
    # The mask should be all 1s.
    assert np.all(mask == 1.0)


def test_runner_completes_if_no_uncertainty(mock_system_config: SystemConfig) -> None:
    """Test the generator completes without yielding if threshold is not met."""
    quantifier = UncertaintyQuantifier()
    quantifier._mock_sequence = [1.0, 2.0, 3.0]  # All grades are below the threshold
    runner = LammpsRunner(
        config=mock_system_config, potential_path="test.yace", quantifier=quantifier
    )
    # The generator should be exhausted without yielding anything
    with pytest.raises(StopIteration):
        next(runner.run())


def test_runner_handles_simulation_error(
    mock_system_config: SystemConfig, mocker: MagicMock
) -> None:
    """Test that LammpsRunner catches and re-raises simulation errors."""
    quantifier = UncertaintyQuantifier()
    mocker.patch.object(quantifier, "get_extrapolation_grade", side_effect=ValueError("Test error"))
    runner = LammpsRunner(
        config=mock_system_config, potential_path="test.yace", quantifier=quantifier
    )
    generator = runner.run()

    with pytest.raises(RuntimeError, match="LAMMPS simulation failed at step 1"):
        next(generator)


def test_runner_stops_at_total_steps(mock_system_config: SystemConfig) -> None:
    """Test that the simulation stops after the specified number of steps."""
    assert mock_system_config.inference is not None  # For type checker
    mock_system_config.inference.total_simulation_steps = 5
    quantifier = UncertaintyQuantifier()
    quantifier._mock_sequence = [1.0] * 10
    runner = LammpsRunner(
        config=mock_system_config, potential_path="test.yace", quantifier=quantifier
    )
    # The generator should exhaust without yielding anything.
    results = list(runner.run())
    assert len(results) == 0
