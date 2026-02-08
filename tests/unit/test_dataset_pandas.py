import gzip
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from mlip_autopipec.core.dataset import Dataset
from mlip_autopipec.domain_models.structure import Structure


def create_labeled_structure() -> Structure:
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10], pbc=True)
    calc = SinglePointCalculator(atoms, energy=-10.5, forces=[[0, 0, 0.1], [0, 0, -0.1]], stress=np.zeros((3, 3)))
    atoms.calc = calc
    return Structure.from_ase(atoms)

def test_dataset_to_pandas(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset = Dataset(dataset_path, root_dir=tmp_path)

    structures = [create_labeled_structure() for _ in range(5)]
    dataset.append(structures)

    # This method should be implemented in Dataset
    # Assuming it returns a pandas DataFrame
    df = dataset.to_pandas()

    assert len(df) == 5
    assert "energy" in df.columns
    assert "forces" in df.columns
    assert "ase_atoms" in df.columns

    # Verify content
    first_row = df.iloc[0]
    assert first_row["energy"] == -10.5
    assert isinstance(first_row["ase_atoms"], Atoms)

def test_dataset_to_pacemaker_gzip(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset = Dataset(dataset_path, root_dir=tmp_path)

    structures = [create_labeled_structure() for _ in range(5)]
    dataset.append(structures)

    output_path = tmp_path / "input.pckl.gzip"
    dataset.to_pacemaker_gzip(output_path)

    assert output_path.exists()

    # Verify it can be read back by pandas (simulating Pacemaker)
    import pandas as pd
    with gzip.open(output_path, "rb") as f:
        df = pd.read_pickle(f)

    assert len(df) == 5
    assert "energy" in df.columns
