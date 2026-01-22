import numpy as np
import re
from ase import Atoms
from mlip_autopipec.dft.inputs import InputGenerator
from mlip_autopipec.dft.constants import SSSP_EFFICIENCY_1_1

def test_kpoint_calculation():
    # Cubic cell 10x10x10. Reciprocal vector length ~ 2pi/10 = 0.628
    # kspacing = 0.1 (1/A).
    # Target implementation: k = ceil(2*pi / (L * kspacing))
    # k = ceil(6.283 / (10 * 0.1)) = ceil(6.283) = 7

    atoms = Atoms(cell=[10, 10, 10], pbc=True)
    kspacing = 0.1 # 1/A

    kpts = InputGenerator._calculate_kpoints(atoms, kspacing)
    assert kpts == (7, 7, 7)

def test_input_generation_flags():
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10], pbc=True)

    # We call create_input_string
    content = InputGenerator.create_input_string(atoms, {"ecutwfc": 40.0})

    # Check flags robustly using regex
    # Match tprnfor = .true. with flexible spacing and case
    assert re.search(r"tprnfor\s*=\s*\.true\.", content, re.IGNORECASE)
    assert re.search(r"tstress\s*=\s*\.true\.", content, re.IGNORECASE)
    assert re.search(r"disk_io\s*=\s*'low'", content, re.IGNORECASE)

def test_magnetism_detection():
    # Fe is magnetic
    atoms = Atoms('Fe', cell=[2.8, 2.8, 2.8], pbc=True)
    content = InputGenerator.create_input_string(atoms)

    assert "nspin" in content and "2" in content
    # Check if starting_magnetization is set (via ASE it might be handled, but checking text is safer)
    # ASE writes starting_magnetization(1) = 2.0 or similar
    assert "starting_magnetization" in content

def test_pseudopotentials():
    atoms = Atoms('Al', cell=[4,4,4], pbc=True)
    content = InputGenerator.create_input_string(atoms)

    # Check if correct pseudo is used (from constants)
    expected_pseudo = SSSP_EFFICIENCY_1_1.get("Al", "Al.UPF")
    assert expected_pseudo in content
