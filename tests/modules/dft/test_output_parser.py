"""Unit tests for the QEOutputParser."""

from pathlib import Path

import pytest

from mlip_autopipec.modules.dft.output_parser import QEOutputParser

# A canonical, complete Quantum Espresso output that ASE can parse
SAMPLE_QE_OUTPUT = """
     Program PWSCF v.6.5 starts on 10Jan2024 at 10:10:10
     bravais-lattice index     = 0
     lattice parameter (a_0)   = 18.897261  a.u.
     celldm(1)=   18.897261
     number of atoms/cell      = 1
     number of atomic types    = 1
     crystal axes: (cart. coord. in units of a_0)
               a(1) = (   0.529177   0.000000   0.000000 )
               a(2) = (   0.000000   0.529177   0.000000 )
               a(3) = (   0.000000   0.000000   0.529177 )
     site n.     atom                  positions (alat units)
         1           Ni          tau(   1) = (   0.0000000   0.0000000   0.0000000  )
!    total energy              =      -1.00000000 Ry
     Forces acting on atoms (Ry/au):

     atom    1 type  1   force =     0.100000000   0.200000000   0.300000000

     Total force =     0.374166     Total SCF correction =     0.000000
     total   stress  (Ry/bohr**3)     (kbar)     P=   -0.00
      0.00000000   0.00000000   0.00000000
      0.00000000   0.00000000   0.00000000
      0.00000000   0.00000000   0.00000000
     JOB DONE.
"""


def test_qeoutputparser_parse_happy_path(tmp_path: Path) -> None:
    """Test the QEOutputParser with a valid QE output file."""
    output_path = tmp_path / "dft.out"
    output_path.write_text(SAMPLE_QE_OUTPUT)

    parser = QEOutputParser()
    results = parser.parse(output_path)

    assert "energy" in results
    assert results["energy"] == pytest.approx(-13.60569)
    assert "forces" in results
    assert results["forces"][0][0] == pytest.approx(0.1 * 13.60569 / 0.529177)
