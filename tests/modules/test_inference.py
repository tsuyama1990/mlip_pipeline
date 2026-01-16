"""
Unit and integration tests for the `LammpsRunner` and the embedding logic
in `mlip_autopipec.modules.inference`.
"""

from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.geometry import find_mic
from pytest_mock import MockerFixture

from mlip_autopipec.config.models import (
    InferenceConfig,
    MDConfig,
    UncertainStructure,
    UncertaintyConfig,
)
from mlip_autopipec.modules.inference import LammpsRunner, extract_embedded_structure


@pytest.fixture
def lammps_config(tmp_path: Path) -> InferenceConfig:
    """Provides a default InferenceConfig for testing."""
    # The potential path must be parsable by the implementation to get symbols
    potential_path = tmp_path / "model-Si_-2024-01-01.yace"
    potential_path.touch()
    # Create a dummy executable to satisfy Pydantic's FilePath validation
    lammps_exe = tmp_path / "lammps"
    lammps_exe.touch()
    return InferenceConfig(
        lammps_executable=lammps_exe,
        potential_path=potential_path,
        md_params=MDConfig(run_duration=10),
        uncertainty_params=UncertaintyConfig(
            threshold=2.5, embedding_cutoff=8.0, masking_cutoff=4.0
        ),
    )


def test_embedding_wraps_atoms_correctly() -> None:
    """
    Tests the periodic embedding logic.

    It creates a large supercell and selects an atom near a periodic
    boundary. It then asserts that the extracted smaller cell correctly
    "wraps" atoms from the other side of the large cell, preserving the
    periodic environment.
    """
    large_cell = bulk("Si", "diamond", a=5.43, cubic=True) * (4, 4, 4)
    center_atom_index = 0  # An atom at the corner
    config = UncertaintyConfig(embedding_cutoff=6.0, masking_cutoff=4.0)

    # Manually find an atom that *should* be wrapped into the embedded cell
    # Atom 0 is at [0, 0, 0]. The cell vector is ~21.72 along x.
    # An atom at the far side of the box would be at ~21.72.
    # The wrap_positions function will bring it close to 0.
    # Let's find an atom near the far x-boundary by finding the max x-coordinate
    far_atom_index = np.argmax(large_cell.positions[:, 0])

    # Execute the embedding
    result = extract_embedded_structure(large_cell, center_atom_index, config)
    embedded_atoms = result.atoms

    # Assertion 1: A wrapped atom is now present in the new cell at a close position
    original_far_pos = large_cell.positions[far_atom_index]

    # Find the position of the 'far_atom' if it were in the embedded cell's coordinates
    # This involves finding the minimum image of the vector from the center atom to the far atom
    find_mic(
        original_far_pos - large_cell.positions[center_atom_index],
        large_cell.cell,
        pbc=True,
    )

    # Assertion 1: The number of atoms in the embedded cell should be greater
    # than a simple cutoff, proving that atoms were wrapped.
    distances_from_center = np.linalg.norm(
        large_cell.positions - large_cell.positions[center_atom_index], axis=1
    )
    n_atoms_in_simple_cutoff = np.sum(distances_from_center < config.embedding_cutoff)
    assert len(embedded_atoms) > n_atoms_in_simple_cutoff

    # Assertion 2: The new cell is periodic
    assert np.all(embedded_atoms.pbc), "The new cell should be periodic."


def test_force_mask_is_correct() -> None:
    """
    Tests that the force mask correctly identifies core and buffer atoms.
    """
    atoms = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
    center_atom_index = 0
    config = UncertaintyConfig(embedding_cutoff=6.0, masking_cutoff=4.0)

    result = extract_embedded_structure(atoms, center_atom_index, config)
    embedded_atoms = result.atoms
    force_mask = result.force_mask

    # Assertion 1: Mask has the correct shape
    assert len(force_mask) == len(embedded_atoms)

    # Assertion 2: Check core and buffer atoms
    center_point = embedded_atoms.get_center_of_mass()
    _, distances = find_mic(
        embedded_atoms.positions - center_point, embedded_atoms.cell, pbc=True
    )

    for i, dist in enumerate(distances):
        if dist < config.masking_cutoff:
            assert force_mask[i] == 1, f"Atom {i} should be a core atom but is not."
        else:
            assert force_mask[i] == 0, f"Atom {i} should be a buffer atom but is not."

    # Assertion 3: There are both core and buffer atoms
    assert 1 in force_mask, "There should be at least one core atom."
    assert 0 in force_mask, "There should be at least one buffer atom."


def test_lammps_script_generation(
    lammps_config: InferenceConfig, tmp_path: Path
) -> None:
    """
    Tests that the generated LAMMPS script contains the correct parameters
    from the InferenceConfig.
    """
    runner = LammpsRunner(inference_config=lammps_config)
    structure_file = tmp_path / "structure.data"
    structure_file.touch()
    potential_file = lammps_config.potential_path
    atoms = bulk("Si")

    # Method under test
    script_path = runner._prepare_lammps_input(
        atoms, tmp_path, structure_file, potential_file
    )

    script_content = script_path.read_text()

    # Assertions to check if config values are correctly written into the script
    assert f"pair_coeff      * * {potential_file.name} Si" in script_content
    assert (
        f"fix             1 all nvt temp {lammps_config.md_params.temperature} "
        f"{lammps_config.md_params.temperature} 0.1"
    ) in script_content
    assert f"timestep        {lammps_config.md_params.timestep}" in script_content
    assert "compute         uncert all pace/extrapol" in script_content
    assert f"run             {lammps_config.md_params.run_duration}" in script_content


@pytest.mark.integration
def test_end_to_end_uncertainty_detection(
    lammps_config: InferenceConfig, tmp_path: Path, mocker: MockerFixture
) -> None:
    """
    Tests the full LammpsRunner.run workflow.
    """
    # --- Setup Mocks ---
    mock_run = mocker.patch("subprocess.run")

    def side_effect_subprocess_run(*args, **kwargs):
        working_dir = Path(kwargs["cwd"])
        # Create a fake trajectory file
        traj_file = working_dir / "dump.custom"
        with traj_file.open("w") as f:
            # Frame at timestep 0
            f.write("ITEM: TIMESTEP\n0\n")
            f.write("ITEM: NUMBER OF ATOMS\n2\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n")
            f.write("ITEM: ATOMS id type x y z\n1 1 0.0 0.0 0.0\n2 1 1.0 1.0 1.0\n")
            # Frame at timestep 10 (the uncertain one)
            f.write("ITEM: TIMESTEP\n10\n")
            f.write("ITEM: NUMBER OF ATOMS\n2\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n")
            f.write("ITEM: ATOMS id type x y z\n1 1 1.0 1.0 1.0\n2 1 2.0 2.0 2.0\n")

        # Create a fake uncertainty file
        uncert_file = working_dir / "uncertainty.dump"
        with uncert_file.open("w") as f:
            f.write("ITEM: TIMESTEP\n0\n")
            f.write("ITEM: NUMBER OF ATOMS\n2\n")
            f.write("ITEM: ATOMS c_uncert[1]\n0.5\n1.0\n")
            f.write("ITEM: TIMESTEP\n10\n")
            f.write("ITEM: NUMBER OF ATOMS\n2\n")
            f.write("ITEM: ATOMS c_uncert[1]\n0.8\n3.0\n")
        return mocker.Mock(returncode=0)

    mock_run.side_effect = side_effect_subprocess_run
    runner = LammpsRunner(inference_config=lammps_config)

    # --- Execute the Method Under Test ---
    initial_structure = bulk("Si") * (2, 1, 1)
    result = runner.run(initial_structure)

    # --- Assertions ---
    mock_run.assert_called_once()
    assert result is not None
    assert isinstance(result, UncertainStructure)
    assert len(result.atoms) < len(initial_structure)
    assert result.metadata["uncertain_timestep"] == 10
    assert result.metadata["uncertain_atom_index_in_original_cell"] is not None
