import numpy as np
import pytest
from ase.build import bulk

from mlip_autopipec.config.models import InferenceConfig, UncertaintyConfig
from mlip_autopipec.modules.inference import (
    LammpsRunner,
    extract_embedded_structure,
)


@pytest.fixture
def mock_inference_config(tmp_path):
    lammps_exec = tmp_path / "lmp"
    potential_path = tmp_path / "potential.yace"
    lammps_exec.touch()
    potential_path.touch()
    config = InferenceConfig(
        lammps_executable=lammps_exec,
        potential_path=potential_path,
        uncertainty_params=UncertaintyConfig(
            embedding_cutoff=8.0, masking_cutoff=4.0, threshold=0.5
        ),
    )
    return config


def test_extract_embedded_structure():
    """
    Unit test for the periodic embedding and force masking logic.
    """
    large_cell = bulk("Si", "diamond", a=5.43, cubic=True) * (4, 4, 4)
    center_atom_index = len(large_cell) // 2  # Pick an atom in the middle
    config = UncertaintyConfig(embedding_cutoff=6.0, masking_cutoff=3.0, threshold=0.5)

    result = extract_embedded_structure(large_cell, center_atom_index, config)

    # Check embedding
    assert result.atoms.get_pbc().all()
    assert np.all(result.atoms.cell.lengths() > 11.9)  # Close to 2*cutoff
    assert len(result.atoms) < len(large_cell)

    # Check force masking
    assert result.force_mask.dtype == int
    assert np.sum(result.force_mask) > 0  # At least one core atom
    assert np.sum(result.force_mask) < len(result.atoms)  # At least one buffer


def test_lammps_script_generation(mock_inference_config, tmp_path):
    """
    Test that the LAMMPS input script is generated correctly.
    """
    runner = LammpsRunner(inference_config=mock_inference_config)
    atoms = bulk("Si", "diamond", a=5.43)
    structure_file = tmp_path / "structure.data"
    script_path = runner._prepare_lammps_input(
        atoms, tmp_path, structure_file, mock_inference_config.potential_path
    )
    script_content = script_path.read_text()

    assert "pair_style      pace" in script_content
    assert f"pair_coeff      * * {mock_inference_config.potential_path.name} Si" in script_content
    assert f"read_data       {structure_file.name}" in script_content
    assert f"timestep        {runner.config.md_params.timestep}" in script_content
    assert f"run             {runner.config.md_params.run_duration}" in script_content


@pytest.mark.integration
def test_end_to_end_uncertainty_detection(mock_inference_config, tmp_path):
    """
    Test the full LAMMPS run and uncertainty detection workflow by mocking
    the subprocess and the LAMMPS output files.
    """
    from unittest.mock import MagicMock, patch

    runner = LammpsRunner(inference_config=mock_inference_config)
    atoms = bulk("Si", "diamond", a=5.43)

    # Create mock LAMMPS output files
    dump_content = "ITEM: TIMESTEP\n10\nITEM: NUMBER OF ATOMS\n2\nITEM: ATOMS id type x y z\n1 1 0.0 0.0 0.0\n2 1 1.35 1.35 1.35\n"
    uncertainty_content = (
        "ITEM: TIMESTEP\n10\nITEM: NUMBER OF ATOMS\n2\nITEM: ATOMS c_uncert[1]\n0.1\n0.8\n"
    )
    (tmp_path / "dump.custom").write_text(dump_content)
    (tmp_path / "uncertainty.dump").write_text(uncertainty_content)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = runner.run(atoms)

    assert result is not None
    assert result.metadata.uncertain_timestep == 10
    assert result.metadata.uncertain_atom_id == 2
    assert result.atoms.get_chemical_symbols() == ["Si", "Si"]
