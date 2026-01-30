import pytest
import numpy as np
from mlip_autopipec.physics.dft.parser import DFTParser
from mlip_autopipec.domain_models.calculation import SCFError, DFTResult

SUCCESS_OUTPUT = """
     Program PWSCF v.6.8 starts on  3Jan2025 at 12:00:00

     total cpu time spent up to now is        0.1 secs

     End of self-consistent calculation

!    total energy              =     -15.00000000 Ry
     Harris-Foulkes estimate   =     -15.00000000 Ry
     estimated scf accuracy    <       0.00000001 Ry

     Forces acting on atoms (cartesian axes, Ry/au):

     atom    1 type  1   force =     0.00000000    0.00000000    0.00000000
     atom    2 type  1   force =     0.10000000    0.00000000    0.00000000

     total   stress  (Ry/bohr**3)                   (kbar)     P=   -0.00
  -0.00000000   0.00000000   0.00000000        -0.00      0.00      0.00
   0.00000000  -0.00000000   0.00000000         0.00     -0.00      0.00
   0.00000000   0.00000000  -0.00000000         0.00      0.00     -0.00

     JOB DONE.
"""

FAILURE_OUTPUT = """
     Program PWSCF v.6.8 starts on  3Jan2025 at 12:00:00

     total cpu time spent up to now is        0.1 secs

     convergence not achieved

     Error in routine c_bands (1):
     too many bands are not converged
"""

def test_parse_success():
    parser = DFTParser()
    # Note: Real parser might need file path or stream.
    # For TDD I assume it can take lines or string or I'll mock file reading.
    # Let's assume parse(lines: Iterator[str]) or similar.
    # Or parse_file(path).
    # I'll use a mocked open or temp file.

    # Actually, simpler is to design parser to take string or list of lines for easier testing.
    result = parser.parse(SUCCESS_OUTPUT)

    assert isinstance(result, DFTResult)
    assert result.converged
    # Energy is parsed in eV usually, PWSCF output is Ry. 1 Ry = 13.6057 eV.
    # But ASE parser handles this. If I use ASE under the hood, I expect eV.
    # If I parse manually, I need to convert.
    # Spec says "Implement parser.py... Parse standard text output...".
    # I should probably rely on ASE's read if possible, but spec implies implementing regex patterns for errors.
    # For successful result, I can delegate to ASE. For errors, I scan.

    # If I delegate to ASE, the units will be eV.
    # -15.0 Ry * 13.6057 ~ -204 eV.
    assert result.energy < -100.0
    assert result.forces.shape == (2, 3)

def test_parse_failure():
    parser = DFTParser()
    with pytest.raises(SCFError):
        parser.parse(FAILURE_OUTPUT)
