from ase import Atoms

from mlip_autopipec.utils.ase_utils import tag_atoms_with_metadata
from mlip_autopipec.utils.qe_utils import (
    get_kpoints,
    get_sssp_recommendations,
)


def test_tag_atoms_with_metadata() -> None:
    """Test tagging an Atoms object with metadata."""
    atoms = Atoms("H")
    metadata = {"key": "value"}
    tagged_atoms = tag_atoms_with_metadata(atoms, metadata)
    assert tagged_atoms.info["key"] == "value"


def test_get_sssp_recommendations() -> None:
    """Test getting SSSP recommendations for an Atoms object."""
    atoms = Atoms("Si")
    recommendations = get_sssp_recommendations(atoms)
    assert "Si" in recommendations
    assert recommendations["Si"].endswith(".UPF")


def test_get_kpoints() -> None:
    """Test calculating k-points for an Atoms object."""
    atoms = Atoms("H", cell=[1, 1, 1], pbc=True)
    kpoints = get_kpoints(atoms)
    assert isinstance(kpoints, tuple)
    assert len(kpoints) == 3
    assert all(isinstance(k, int) for k in kpoints)
