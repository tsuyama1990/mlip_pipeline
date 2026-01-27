from ase import Atoms
from ase.build import bulk

from mlip_autopipec.data_models.dft_models import DFTInputParams
from mlip_autopipec.dft.inputs import InputGenerator


def test_create_input_string_basic():
    atoms = bulk("Al", "fcc", a=4.05)
    params = DFTInputParams(
        ecutwfc=30.0, kspacing=0.04, mixing_beta=0.4, smearing="mv", degauss=0.01
    )

    input_str = InputGenerator.create_input_string(atoms, params)
    upper_str = input_str.upper()

    assert "&CONTROL" in upper_str
    assert "CALCULATION" in upper_str
    assert "SCF" in upper_str
    assert "TSTRESS" in upper_str
    assert "ECUTWFC" in upper_str
    assert "MIXING_BETA" in upper_str
    assert "ATOMIC_POSITIONS" in upper_str
    assert "K_POINTS" in upper_str


def test_create_input_string_magnetism():
    atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=True)
    input_str = InputGenerator.create_input_string(atoms)
    upper_str = input_str.upper()

    assert "NSPIN" in upper_str
    # Fe is in MAGNETIC_ELEMENTS usually


def test_kpoints_calculation():
    atoms = bulk("Si", "diamond", a=5.43)

    kpts = InputGenerator._calculate_kpoints(atoms, kspacing=0.5)
    assert all(k >= 1 for k in kpts)

    supercell = atoms * (2, 2, 2)
    kpts_super = InputGenerator._calculate_kpoints(supercell, kspacing=0.5)
    # Larger cell should have fewer or equal kpoints
    assert all(ks <= k for ks, k in zip(kpts_super, kpts, strict=False))


def test_pseudopotentials():
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    pseudos = InputGenerator._get_pseudopotentials(atoms)

    assert "H" in pseudos
    assert "O" in pseudos
    assert pseudos["H"].endswith(".UPF")
    assert pseudos["O"].endswith(".UPF")
