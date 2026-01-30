from pathlib import Path
import pytest
import numpy as np

from mlip_autopipec.domain_models.calculation import DFTResult, SCFError, MemoryError, WalltimeError
from mlip_autopipec.physics.dft.parser import DFTParser

# Sample outputs (shortened for brevity)

SUCCESS_OUTPUT = """
     Program PWSCF v.6.8 starts on 12Jan2025 at 12:00:00
     ...
     !    total energy              =     -15.00000000 Ry
     ...
     Forces acting on atoms (cartesian axes, Ry/au):

     atom    1 type  1   force =     0.00000000    0.00000000    0.00000000
     atom    2 type  1   force =     0.00000000    0.00000000    0.00000000

     total   stress  (Ry/bohr**3)                   (kbar)     P=   -0.00
  -0.00000000   0.00000000   0.00000000        -0.00      0.00      0.00
   0.00000000  -0.00000000   0.00000000         0.00     -0.00      0.00
   0.00000000   0.00000000  -0.00000000         0.00      0.00     -0.00
     ...
     JOB DONE.
"""

SCF_FAILURE_OUTPUT = """
     ...
     total energy              =     -14.90000000 Ry
     estimated scf accuracy    <       0.10000000 Ry

     convergence not achieved
     ...
"""

MEMORY_FAILURE_OUTPUT = """
     ...
     feature not implemented
     %!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     %!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     %!!                                                                          !!
     %!!     Error in routine diropn (1):                                         !!
     %!!     error opening /scratch/tmp.save/wfc1.dat                             !!
     %!!                                                                          !!
     %!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     %!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     ...
     Maximum CPU time exceeded
"""
# Actually "Maximum CPU time exceeded" is Walltime. "error opening" could be disk/memory.
# "Out Of Memory" or "SIGKILL" usually comes from scheduler, but QE might print "allocation failed".

WALLTIME_FAILURE_OUTPUT = """
     ...
     Maximum CPU time exceeded
     ...
"""

# Helper to create a dummy file
@pytest.fixture
def create_dummy_file(tmp_path):
    def _create(content, filename="pw.out"):
        p = tmp_path / filename
        p.write_text(content)
        return p
    return _create

def test_parse_success(create_dummy_file):
    # We need a parser that can read the file.
    # However, for parsing values (energy, forces), we might rely on ASE `read_espresso_out`.
    # But ASE needs a valid formatted file.
    # So I will test mainly the error detection logic here,
    # and if I mock ASE reading, I can test the result creation.

    # Actually, the parser should first check for errors.
    # If no errors, call ASE.

    # Since I cannot easily create a full valid QE output that ASE accepts without running QE,
    # I will mock `ase.io.read`.

    from unittest.mock import patch, MagicMock
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    path = create_dummy_file(SUCCESS_OUTPUT)

    # Mock ASE read
    atoms = Atoms(symbols=["Si", "Si"], positions=[[0,0,0], [1,1,1]])
    calc = SinglePointCalculator(
        atoms,
        energy=-204.0, # eV
        forces=np.zeros((2, 3)),
        stress=np.zeros(6), # Voigt
        magmoms=None
    )
    atoms.calc = calc

    with patch("ase.io.read", return_value=atoms) as mock_read:
        result = DFTParser.parse(path, job_id="test_job", work_dir=path.parent, duration=100.0)

        assert isinstance(result, DFTResult)
        assert result.energy == -204.0
        assert result.forces.shape == (2, 3)
        assert result.status.value == "COMPLETED"

def test_detect_scf_error(create_dummy_file):
    path = create_dummy_file(SCF_FAILURE_OUTPUT)

    with pytest.raises(SCFError):
        DFTParser.parse(path, job_id="test_job", work_dir=path.parent, duration=100.0)

def test_detect_walltime_error(create_dummy_file):
    path = create_dummy_file(WALLTIME_FAILURE_OUTPUT)

    with pytest.raises(WalltimeError):
        DFTParser.parse(path, job_id="test_job", work_dir=path.parent, duration=100.0)

def test_detect_memory_error(create_dummy_file):
    # Construct a memory error string that ASE might not catch but we should.
    # Common QE error: "from allocate_nlpot : error # 1 : cannot allocate"
    output = " ... \n %!! error # 1 : cannot allocate ... \n"
    path = create_dummy_file(output)

    with pytest.raises(MemoryError):
        DFTParser.parse(path, job_id="test_job", work_dir=path.parent, duration=100.0)
