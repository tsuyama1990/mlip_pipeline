from ase import Atoms

from mlip_autopipec.dft.input_gen import write_pw_input


def test_write_pw_input():
    atoms = Atoms("Si")
    params = {"control": {"calculation": "scf"}, "system": {"ecutwfc": 30}}
    pseudos = {"Si": "Si.upf"}
    kpts = (1, 1, 1)

    txt = write_pw_input(atoms, params, pseudos, kpts)

    # Check for presence of key parameters, handling variable whitespace
    assert "calculation" in txt
    assert "'scf'" in txt
    assert "tprnfor" in txt
    assert ".true." in txt
    assert "tstress" in txt
    assert ".true." in txt
    assert "disk_io" in txt
    assert "'low'" in txt
    assert "Si.upf" in txt
