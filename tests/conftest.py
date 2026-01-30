
import ase
import pytest
from mlip_autopipec.domain_models.structure import Structure


@pytest.fixture
def sample_ase_atoms() -> ase.Atoms:
    return ase.Atoms(
        symbols=["H", "H"],
        positions=[[0, 0, 0], [0, 0, 0.74]],
        cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
        pbc=[True, True, True],
        info={"energy": -1.5}
    )

@pytest.fixture
def minimal_structure(sample_ase_atoms: ase.Atoms) -> Structure:
    return Structure.from_ase(sample_ase_atoms)
