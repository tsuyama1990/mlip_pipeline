# ruff: noqa: D101, D102, D103
"""Tests for the inference module."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from ase.build import bulk

from mlip_autopipec.modules.inference import LammpsRunner, UncertaintyQuantifier


@pytest.fixture
def mock_config():
    """Return a mock SystemConfig object for inference."""
    config = MagicMock()
    config.inference.uncertainty_threshold = 3.0
    config.inference.embedding_cutoff = 5.0
    config.inference.embedding_buffer = 1.0
    config.inference.total_simulation_steps = 10
    return config


@pytest.fixture
def lammps_runner(mock_config):
    """Return an instance of the LammpsRunner."""
    quantifier = UncertaintyQuantifier()
    return LammpsRunner(mock_config, "potential.yace", quantifier)


def test_extract_periodic_subcell(lammps_runner):
    """Test the extraction of a periodic sub-cell."""
    atoms = bulk("Cu", "fcc", a=3.6) * (5, 5, 5)
    center_atom_index = len(atoms) // 2

    # Replace the placeholder with a real implementation for testing
    def _extract_real(atoms, center_atom_index):
        cutoff = lammps_runner.config.inference.embedding_cutoff
        buffer = lammps_runner.config.inference.embedding_buffer
        box_size = 2 * (cutoff + buffer)

        center_pos = atoms.positions[center_atom_index]

        # This is a simplified logic, a real one would handle PBC wrapping
        distances = np.linalg.norm(atoms.positions - center_pos, axis=1)
        indices_in_box = np.where(distances < box_size / 2)[0]

        sub_cell = atoms[indices_in_box].copy()
        sub_cell.set_cell([box_size, box_size, box_size])
        sub_cell.pbc = True
        return sub_cell

    lammps_runner._extract_periodic_subcell = _extract_real

    sub_cell = lammps_runner._extract_periodic_subcell(atoms, center_atom_index)

    assert sub_cell is not None
    assert all(sub_cell.pbc)
    assert len(sub_cell) < len(atoms)
    assert np.all(sub_cell.cell.lengths() == (2 * (5.0 + 1.0)))


def test_generate_force_mask(lammps_runner):
    """Test the generation of a force mask."""
    box_size = 12.0
    atoms = bulk("Cu", "fcc", a=3.6)
    atoms.set_cell([box_size, box_size, box_size])
    atoms.center()

    # Replace the placeholder with a real implementation
    def _generate_real(atoms):
        cutoff = lammps_runner.config.inference.embedding_cutoff
        center = np.diag(atoms.cell) / 2.0
        distances = np.linalg.norm(atoms.positions - center, axis=1)
        mask = (distances < cutoff).astype(float)
        return mask

    lammps_runner._generate_force_mask = _generate_real

    mask = lammps_runner._generate_force_mask(atoms)

    assert mask is not None
    assert mask.shape == (len(atoms),)
    # The center atom should be inside the cutoff
    assert mask[0] == 1.0


def test_lammps_runner_yields_masked_structure_on_high_uncertainty(lammps_runner):
    """Test that the runner yields a sub-cell and mask on high uncertainty."""
    # Mock the uncertainty to be high on the first step
    lammps_runner.quantifier.get_extrapolation_grade = MagicMock(return_value=4.5)

    # Mock the helper methods to return predictable values
    mock_sub_cell = bulk("Cu")
    mock_mask = np.ones(1)
    lammps_runner._extract_periodic_subcell = MagicMock(return_value=mock_sub_cell)
    lammps_runner._generate_force_mask = MagicMock(return_value=mock_mask)

    simulation_generator = lammps_runner.run()
    atoms, grade, force_mask = next(simulation_generator)

    assert atoms is mock_sub_cell
    assert grade == 4.5
    assert force_mask is mock_mask


def test_lammps_runner_yields_full_structure_on_low_uncertainty(lammps_runner):
    """Test that the runner yields the full cell and no mask on low uncertainty."""
    # Mock the uncertainty to be low
    lammps_runner.quantifier.get_extrapolation_grade = MagicMock(return_value=1.5)

    simulation_generator = lammps_runner.run()
    atoms, grade, force_mask = next(simulation_generator)

    # The mock cell is 4x4x4 = 64 atoms
    assert len(atoms) == 64
    assert grade == 1.5
    assert force_mask is None
