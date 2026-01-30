import pytest
from mlip_autopipec.physics.dft.parser import DFTParser
from mlip_autopipec.domain_models.calculation import SCFError, DFTResult, WalltimeError, DFTError

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

WALLTIME_OUTPUT = """
     Program PWSCF v.6.8 starts on ...
     maximum CPU time exceeded
     stopping ...
"""

MEMORY_OUTPUT = """
     Program PWSCF v.6.8 starts on ...

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     Error in routine diropn (1):
     input/output error
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     stopping ...
"""
# Note: "Out of memory" is not a standard QE message, often from scheduler/system.
# But QE might print allocation errors.
ALLOC_OUTPUT = """
     Error in routine allocate_nlpot (1):
     failed to allocate
"""

def test_parse_success():
    parser = DFTParser()
    result = parser.parse(SUCCESS_OUTPUT)

    assert isinstance(result, DFTResult)
    assert result.converged
    assert result.energy < -100.0
    assert result.forces.shape == (2, 3)

def test_parse_failure():
    parser = DFTParser()
    with pytest.raises(SCFError):
        parser.parse(FAILURE_OUTPUT)

def test_parse_walltime():
    parser = DFTParser()
    with pytest.raises(WalltimeError):
        parser.parse(WALLTIME_OUTPUT)

def test_parse_general_error():
    # Test a generic error not SCF
    output = """
     Error in routine something (1):
     bad things happened
    """
    parser = DFTParser()
    with pytest.raises(DFTError):
        parser.parse(output)
