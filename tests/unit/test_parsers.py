from pathlib import Path

import pytest
from ase.units import Bohr, Ry

from mlip_autopipec.utils.parsers import DFTConvergenceError, QEParser


def test_parse_converged_output() -> None:
    with Path("tests/data/qe_outputs/converged.out").open() as f:
        content = f.read()

    parser = QEParser(content)

    # Check convergence (should not raise)
    parser.check_convergence()

    # Energy
    # !    total energy              =     -156.23456789 Ry
    expected_energy = -156.23456789 * Ry
    assert parser.parse_energy() == pytest.approx(expected_energy)

    # Forces
    # atom    1 type  1   force =     0.10000000    0.00000000    0.00000000
    forces = parser.parse_forces()
    assert len(forces) == 2
    expected_force_x = 0.1 * (Ry / Bohr)
    assert forces[0][0] == pytest.approx(expected_force_x)
    assert forces[0][1] == 0.0

    # Stress
    # -0.00000010   0.00000000   0.00000000
    stress = parser.parse_stress()
    assert stress is not None
    assert len(stress) == 3
    expected_stress_xx = -0.00000010 * (Ry / Bohr**3)
    assert stress[0][0] == pytest.approx(expected_stress_xx)


def test_parse_convergence_error() -> None:
    with Path("tests/data/qe_outputs/scf_error.out").open() as f:
        content = f.read()

    parser = QEParser(content)

    with pytest.raises(DFTConvergenceError, match="SCF convergence not achieved"):
        parser.check_convergence()


def test_parse_missing_energy() -> None:
    parser = QEParser("Some junk output")
    with pytest.raises(ValueError, match="Could not find energy"):
        parser.parse_energy()


def test_parse_missing_forces() -> None:
    parser = QEParser("Some junk output")
    assert parser.parse_forces() == []


def test_parse_missing_stress() -> None:
    parser = QEParser("Some junk output")
    assert parser.parse_stress() is None
