import pytest
from ase import Atoms
from pathlib import Path
from mlip_autopipec.dft.input_gen import write_pw_input

def test_write_pw_input(tmp_path: Path) -> None:
    atoms = Atoms("Si2", positions=[[0,0,0], [0.25, 0.25, 0.25]], cell=[5.43]*3, pbc=True)
    filename = tmp_path / "pw.in"
    input_data = {
        "control": {"calculation": "scf", "restart_mode": "from_scratch"},
        "system": {"ecutwfc": 30},
        "electrons": {"conv_thr": 1e-6}
    }
    pseudos = {"Si": "Si.upf"}
    kpts = [2, 2, 2]

    write_pw_input(atoms, filename, input_data, pseudos, kpts=kpts)

    import re
    content = filename.read_text()
    # Check for content without being too strict on whitespace
    assert "calculation" in content
    assert "scf" in content
    assert "disk_io" in content
    assert "low" in content
    assert "tprnfor" in content
    assert ".true." in content
    assert "tstress" in content
    # .true. is already checked
    assert "K_POINTS automatic" in content
    assert re.search(r"2\s+2\s+2\s+0\s+0\s+0", content)

def test_write_pw_input_defaults(tmp_path: Path) -> None:
    # Test handling of None for optional args
    atoms = Atoms("H")
    filename = tmp_path / "pw_defaults.in"
    input_data = {"control": {}}
    pseudos = {"H": "H.upf"}

    write_pw_input(atoms, filename, input_data, pseudos)

    content = filename.read_text()
    assert "calculation" in content # Should default to scf
    assert "tprnfor" in content
    assert "K_POINTS gamma" in content # Default if kpts=None

def test_write_pw_input_custom_koffset(tmp_path: Path) -> None:
    atoms = Atoms("H")
    filename = tmp_path / "pw_offset.in"
    input_data = {}
    pseudos = {"H": "H.upf"}
    kpts = [2, 2, 2]
    koffset = [1, 1, 1]

    write_pw_input(atoms, filename, input_data, pseudos, kpts=kpts, koffset=koffset)

    content = filename.read_text()
    import re
    assert re.search(r"2\s+2\s+2\s+1\s+1\s+1", content)
