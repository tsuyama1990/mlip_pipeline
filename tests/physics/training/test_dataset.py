import pytest
from ase import Atoms
import pandas as pd
import numpy as np
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.training import dataset


def test_atoms_to_dataframe():
    # Create dummy structures
    atoms1 = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[10, 10, 10], pbc=True)
    atoms1.info["energy"] = -10.0
    atoms1.info["stress"] = np.eye(3).flatten()
    atoms1.set_array("forces", np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]))

    structure1 = Structure.from_ase(atoms1)

    # We create a list
    structures = [structure1]

    # Call the function
    df = dataset.atoms_to_dataframe(structures)

    # Verify
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert not df.empty
    assert "ase_atoms" in df.columns
    assert "energy" in df.columns
    assert df.iloc[0]["energy"] == -10.0

    # Check if ase_atoms has forces
    reconstructed_atoms = df.iloc[0]["ase_atoms"]
    assert "forces" in reconstructed_atoms.arrays
    np.testing.assert_array_equal(reconstructed_atoms.arrays["forces"], atoms1.arrays["forces"])
