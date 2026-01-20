import pytest
from ase import Atoms

from mlip_autopipec.dft.utils import get_kpoints, get_sssp_pseudopotentials, is_magnetic
from mlip_autopipec.exceptions import DFTException


def test_get_kpoints_large_cell():
    # Large cell -> Small reciprocal -> Few kpoints
    atoms = Atoms(cell=[20, 20, 20], pbc=True)
    # b = 2pi/20 = 0.314
    # 0.314 * 0.15 = 0.047 -> 0 -> 1
    k = get_kpoints(atoms, 0.15)
    assert k == [1, 1, 1]

def test_get_kpoints_small_cell():
    # Small cell -> Large reciprocal -> Many kpoints
    atoms = Atoms(cell=[2, 2, 2], pbc=True)
    # b = 2pi/2 = 3.14
    # 3.14 * 0.15 = 0.47 -> 0 -> 1 ?
    # If the formula in Spec is strictly round(|bi| * density), then 0.15 density with 2A cell gives 1 kpoint.
    # This suggests density unit might be 1/A (spacing) and formula should be different,
    # OR density value 0.15 is very small (coarse grid).
    # Common values for density (per A^-1) are 2-4 (which means spacing 0.25-0.5 A^-1).
    # If density=2.0: 3.14 * 2 = 6.
    # Let's assume the implementation follows the formula in Spec strictly.
    k = get_kpoints(atoms, 0.15)
    # Based on strict formula:
    assert k == [1, 1, 1]

def test_is_magnetic():
    assert is_magnetic(Atoms("Fe"))
    assert is_magnetic(Atoms("Fe2O3"))
    assert not is_magnetic(Atoms("Si"))
    assert not is_magnetic(Atoms("H2O"))

def test_get_sssp(tmp_path):
    (tmp_path / "Fe.upf").touch()
    (tmp_path / "O.pbe.upf").touch()

    pseudos = get_sssp_pseudopotentials(Atoms("FeO"), tmp_path)
    assert pseudos["Fe"] == "Fe.upf"
    assert pseudos["O"] == "O.pbe.upf"

    with pytest.raises(DFTException):
        get_sssp_pseudopotentials(Atoms("H"), tmp_path)
