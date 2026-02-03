import io
from typing import ClassVar

import pytest
from ase import Atoms

from mlip_autopipec.inference.pace_driver import print_results, read_geometry

# Sample EON input (assuming simple XYZ-like or specific EON format)
# Format:
# N_atoms
# Energy
# Box1
# Box2
# Box3
# Atom1
# ...
SAMPLE_INPUT = """2
0.0
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0
Cu 0.0 0.0 0.0
Cu 2.0 0.0 0.0
"""

class MockResults:
    energy = -10.5
    forces: ClassVar[list[list[float]]] = [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]
    stress: ClassVar[list[float]] = [0.0] * 6
    gamma = 0.0

def test_read_geometry() -> None:
    stream = io.StringIO(SAMPLE_INPUT)
    atoms = read_geometry(stream)
    assert isinstance(atoms, Atoms)
    assert len(atoms) == 2
    assert atoms.get_chemical_symbols() == ["Cu", "Cu"]  # type: ignore[no-untyped-call]

def test_print_results(capsys: pytest.CaptureFixture[str]) -> None:
    results = MockResults()
    print_results(results)
    captured = capsys.readouterr()
    # Check that energy and forces are in output
    # Output format is scientific notation
    assert "-1.05" in captured.out  # -10.5 is -1.05e+01
    assert "1.000" in captured.out  # 0.1 is 1.000e-01
