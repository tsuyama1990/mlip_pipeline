import pytest
import numpy as np
from pathlib import Path
from mlip_autopipec.physics.dft.parser import DFTParser
from mlip_autopipec.domain_models.calculation import DFTResult, SCFError
from mlip_autopipec.domain_models.job import JobStatus

SUCCESS_OUTPUT = """
     Program PWSCF v.6.8 starts on  3Feb2025 at 12:00:00

     total cpu time spent up to now is        1.2 secs

     End of self-consistent calculation

!    total energy              =     -15.78901234 Ry
     Harris-Foulkes estimate   =     -15.78901234 Ry
     estimated scf accuracy    <       0.00000001 Ry

     Forces acting on atoms (cartesian axes, Ry/au):

     atom    1 type  1   force =     0.00000000    0.00000000    0.00120000
     atom    2 type  1   force =     0.00000000    0.00000000   -0.00120000

     stress (Ry/bohr**3)                   (kbar)     P=   -0.50
    -0.00000330   0.00000000   0.00000000        -0.49      0.00      0.00
     0.00000000  -0.00000330   0.00000000         0.00     -0.49      0.00
     0.00000000   0.00000000  -0.00000350         0.00      0.00     -0.52

     JOB DONE.
"""

FAILURE_OUTPUT = """
     Program PWSCF v.6.8 starts on  3Feb2025 at 12:00:00

     convergence not achieved

     Error in routine c_bands (1):
     too many bands are not converged
"""

def test_parse_success():
    parser = DFTParser()
    result = parser.parse(SUCCESS_OUTPUT, job_id="test", work_dir=Path("."))

    assert isinstance(result, DFTResult)
    assert result.status == JobStatus.COMPLETED
    assert result.energy == pytest.approx(-15.78901234 * 13.6056980659) # Ry to eV
    assert result.forces.shape == (2, 3)
    # Check force conversion (Ry/au -> eV/A)
    # 1 Ry/au = 13.605... / 0.529177... approx 25.711
    # 0.0012 * 25.711 ~ 0.03
    assert np.allclose(result.forces[0], [0, 0, 0.0012 * 13.6056980659 / 0.52917721067], atol=1e-3)

def test_parse_scf_failure():
    parser = DFTParser()
    with pytest.raises(SCFError):
        parser.parse(FAILURE_OUTPUT, job_id="test", work_dir=Path("."))
