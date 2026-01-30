from pathlib import Path
import pytest
import ase
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.calculation import DFTConfig
# Note: InputGenerator is not implemented yet, so we mock or expect import error if we were running it now.
# But for TDD, we write the test assuming the interface.
from mlip_autopipec.physics.dft.input_gen import InputGenerator

def test_input_generation_kpoints():
    atoms = ase.Atoms(
        symbols=["Si"],
        positions=[[0, 0, 0]],
        cell=[[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]],
        pbc=[True, True, True]
    )
    structure = Structure.from_ase(atoms)

    config = DFTConfig(
        command="pw.x",
        pseudopotentials={"Si": Path("Si.upf")},
        ecutwfc=40.0,
        kspacing=0.2  # Should give roughly 2*pi / (5.43 * 0.2) ~ 5 or 6
    )

    generator = InputGenerator(config)
    input_str = generator.generate(structure)

    assert "ATOMIC_SPECIES" in input_str
    assert "Si" in input_str
    assert "Si.upf" in input_str
    assert "K_POINTS automatic" in input_str
    # 2*pi / (5.43 * 0.2) = 6.28 / 1.086 = 5.78 -> 6
    # ASE default for koffset is often 0 0 0 if we passed it.

    # Robust check
    lines = input_str.splitlines()
    k_idx = -1
    for i, line in enumerate(lines):
        if "K_POINTS automatic" in line:
            k_idx = i
            break

    assert k_idx != -1
    k_line = lines[k_idx + 1].strip()
    # It should match "6 6 6 0 0 0" (normalize spaces)
    import re
    k_line_norm = re.sub(r'\s+', ' ', k_line)
    assert k_line_norm == "6 6 6 0 0 0" or k_line_norm == "5 5 5 0 0 0", f"Got K-points line: '{k_line}'"

def test_input_generation_control_flags():
    atoms = ase.Atoms("H2", positions=[[0,0,0], [0,0,0.7]], cell=[10,10,10], pbc=True)
    structure = Structure.from_ase(atoms)

    config = DFTConfig(
        command="pw.x",
        pseudopotentials={"H": Path("H.upf")},
        ecutwfc=30.0,
        tprnfor=True,
        tstress=True
    )

    generator = InputGenerator(config)
    input_str = generator.generate(structure)

    # Check with regex or stripped spaces
    import re
    assert re.search(r"tprnfor\s*=\s*\.true\.", input_str)
    assert re.search(r"tstress\s*=\s*\.true\.", input_str)
    assert re.search(r"calculation\s*=\s*'scf'", input_str)
