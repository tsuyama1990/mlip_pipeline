import gzip
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from mlip_autopipec.core.dataset import Dataset
from mlip_autopipec.domain_models.structure import Structure


def create_labeled_structure() -> Structure:
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10], pbc=True)
    calc = SinglePointCalculator(atoms, energy=-10.5, forces=[[0, 0, 0.1], [0, 0, -0.1]], stress=np.zeros((3, 3)))  # type: ignore[no-untyped-call]
    atoms.calc = calc
    return Structure.from_ase(atoms)

def test_dataset_export_to_extxyz(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset = Dataset(dataset_path, root_dir=tmp_path)

    structures = [create_labeled_structure() for _ in range(5)]
    dataset.append(structures)

    output_path = tmp_path / "dataset.extxyz"
    dataset.to_extxyz(output_path)

    assert output_path.exists()

    # Verify content using ASE
    from ase.io import read
    atoms_list = read(output_path, index=":")
    assert len(atoms_list) == 5

    # ASE extxyz reader puts energy in atoms.info if it's in the comment line
    # or atoms.calc if reading with specific calculator logic
    # Standard ase write/read roundtrip usually preserves info['energy']

    # Check both potential locations
    first_atom = atoms_list[0]
    # Check if first_atom is actually an Atoms object to satisfy MyPy
    if isinstance(first_atom, Atoms):
        energy = first_atom.info.get("energy")
        if energy is None and first_atom.calc:
            energy = first_atom.get_potential_energy()  # type: ignore[no-untyped-call]

        assert energy == -10.5
    else:
        pytest.fail("Read object is not an ASE Atoms object")

def test_dataset_export_to_pacemaker_gzip(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset = Dataset(dataset_path, root_dir=tmp_path)

    structures = [create_labeled_structure() for _ in range(5)]
    dataset.append(structures)

    output_path = tmp_path / "input.pckl.gzip"
    dataset.to_pacemaker_gzip(output_path, chunk_size=2)  # Small chunk size

    assert output_path.exists()

    # Verify it can be read back (simulating Pacemaker)
    # Using standard pickle instead of pd.read_pickle
    with gzip.open(output_path, "rb") as f:
        df = pickle.load(f)  # noqa: S301

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert "energy" in df.columns
