from pathlib import Path

import pytest
from ase import Atoms

from mlip_autopipec.dft.utils import get_kpoints, get_sssp_pseudopotentials, is_magnetic


def test_kpoints_generation() -> None:
    # 10x10x10 cubic cell
    atoms_large = Atoms("H", cell=[10, 10, 10], pbc=True)
    # 2x2x2 cubic cell
    atoms_small = Atoms("H", cell=[2, 2, 2], pbc=True)

    # We inferred density ~ 1.6 to match [5,5,5] for 2A cell.
    # Recip |b| (2A) = 2pi/2 = 3.14. 3.14 * 1.6 = 5.024 -> 5.
    # Recip |b| (10A) = 2pi/10 = 0.628. 0.628 * 1.6 = 1.00 -> 1.
    density = 1.6

    k_large = get_kpoints(atoms_large, density)
    k_small = get_kpoints(atoms_small, density)

    assert k_large == [1, 1, 1]
    assert k_small == [5, 5, 5]


def test_is_magnetic() -> None:
    assert is_magnetic(Atoms("Fe"))
    assert is_magnetic(Atoms("Co"))
    assert is_magnetic(Atoms("Ni"))
    assert is_magnetic(Atoms("Fe2O3"))
    assert not is_magnetic(Atoms("Si"))
    assert not is_magnetic(Atoms("H2O"))


def test_sssp_mapping(tmp_path: Path) -> None:
    pseudo_dir = tmp_path / "pseudo"
    pseudo_dir.mkdir()
    (pseudo_dir / "Fe.upf").write_text("<UPF version='2.0.1'>Fake content</UPF>")
    (pseudo_dir / "O_pbe.upf").write_text("<UPF version='2.0.1'>Fake content</UPF>")

    mapping = get_sssp_pseudopotentials(["Fe", "O"], pseudo_dir)
    assert mapping["Fe"] == "Fe.upf"
    assert mapping["O"] == "O_pbe.upf"

    with pytest.raises(FileNotFoundError):
        get_sssp_pseudopotentials(["Al"], pseudo_dir)
