"""
Tests for input generation.
"""
import pytest
from ase import Atoms
from mlip_autopipec.dft import input_gen

def test_write_pw_input(tmp_path):
    """Test Quantum Espresso input file generation."""
    atoms = Atoms("Si")
    parameters = {
        "control": {
            "restart_mode": "from_scratch",
            "calculation": "scf"
        }
    }
    pseudos = {"Si": "Si.UPF"}
    kpts = [2, 2, 2]
    output_path = tmp_path / "pw.in"

    input_gen.write_pw_input(atoms, parameters, pseudos, kpts, output_path)

    content = output_path.read_text().lower()

    # Check for mandatory flags
    assert "calculation" in content and "scf" in content
    assert "disk_io" in content and "low" in content

    # ASE might write .true. or .TRUE. or true depending on version
    # Usually .true.
    assert "tprnfor" in content and ".true." in content
    assert "tstress" in content and ".true." in content

    # Check pseudopotentials
    assert "si.upf" in content

    # Check K-points
    # ASE writes "K_POINTS automatic\n2 2 2 0 0 0" or similar
    assert "2 2 2" in content
