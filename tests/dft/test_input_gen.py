from pathlib import Path

from ase import Atoms

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
