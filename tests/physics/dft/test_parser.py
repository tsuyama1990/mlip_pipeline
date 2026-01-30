import pytest
import numpy as np

from mlip_autopipec.domain_models.calculation import SCFError, WalltimeError, MemoryError, DFTError
from mlip_autopipec.physics.dft.parser import DFTParser

SUCCESS_OUTPUT = """
     Program PWSCF v.7.0 starts on  1Jan2025 at 00:00:00

     total cpu time spent up to now is        1.2 secs

     End of self-consistent calculation

     !    total energy              =     -150.12345678 Ry
          Harris-Foulkes estimate   =     -150.12345678 Ry
          estimated scf accuracy    <        1.2E-10 Ry

     Forces acting on atoms (cartesian axes, Ry/au):

     atom    1 type  1   force =     0.00100000    0.00200000    0.00300000
     atom    2 type  1   force =    -0.00100000   -0.00200000   -0.00300000

     Total force =     0.005385     Total SCF correction =     0.000000

     stress (Ry/bohr**3)                   (kbar)     P=   -0.12
  -0.00010000   0.00000000   0.00000000       -14.70        0.00        0.00
   0.00000000  -0.00010000   0.00000000         0.00      -14.70        0.00
   0.00000000   0.00000000  -0.00010000         0.00        0.00      -14.70

     JOB DONE.
"""

SCF_FAIL_OUTPUT = """
     Program PWSCF v.7.0 starts on ...

     convergence not achieved

     Error in routine c_bands (1):
     too many bands
"""


def test_parser_success():
    parser = DFTParser()
    results = parser.parse_output(SUCCESS_OUTPUT)

    # Energy in Ry -> eV
    # -150.12345678 * 13.6056980659
    assert np.isclose(results["energy"], -150.12345678 * 13.6056980659)

    forces = results["forces"]
    assert forces.shape == (2, 3)
    # Ry/au -> eV/A = 13.605... / 0.529177... ~= 25.711
    assert np.allclose(forces[0], np.array([0.001, 0.002, 0.003]) * 25.71104309)

    stress = results["stress"]
    assert stress.shape == (3, 3)


def test_parser_scf_failure():
    parser = DFTParser()
    with pytest.raises(SCFError):
        parser.parse_output(SCF_FAIL_OUTPUT)

def test_parser_walltime_error():
    parser = DFTParser()
    out = "some output... maximum CPU time exceeded ... ending"
    with pytest.raises(WalltimeError):
        parser.parse_output(out)

def test_parser_memory_error():
    parser = DFTParser()
    err = "some error... Out Of Memory ... ending"
    with pytest.raises(MemoryError):
        parser.parse_output("output", stderr=err)

def test_parser_incomplete():
    parser = DFTParser()
    # No JOB DONE, no energy
    with pytest.raises(DFTError, match="Job incomplete"):
        parser.parse_output("Starts... ends abruptly")

def test_parser_job_done_no_energy():
    parser = DFTParser()
    with pytest.raises(DFTError, match="Job done but energy not found"):
        parser.parse_output("Program starts... JOB DONE.")
