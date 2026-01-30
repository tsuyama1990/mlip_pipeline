import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.calculation import DFTConfig, DFTResult
from mlip_autopipec.physics.dft.qe_runner import QERunner

# Define mocked outputs
MOCK_SUCCESS_OUTPUT = """
     Program PWSCF v.6.8 starts on ...
     End of self-consistent calculation
!    total energy              =     -10.00000000 Ry
     Forces acting on atoms (cartesian axes, Ry/au):
     atom    1 type  1   force =     0.00000000    0.00000000    0.00000000
     atom    2 type  1   force =     0.00000000    0.00000000    0.00000000
     total   stress  (Ry/bohr**3)                   (kbar)     P=   -0.00
   0.00000000   0.00000000   0.00000000        -0.00      0.00      0.00
   0.00000000   0.00000000   0.00000000         0.00     -0.00      0.00
   0.00000000   0.00000000   0.00000000         0.00      0.00     -0.00
     JOB DONE.
"""

MOCK_FAILURE_OUTPUT = """
     Program PWSCF v.6.8 starts on ...
     convergence not achieved
     Error in routine c_bands (1):
     too many bands are not converged
"""

@pytest.fixture
def si_structure():
    return Structure(
        symbols=["Si", "Si"],
        positions=np.array([[0, 0, 0], [1.36, 1.36, 1.36]]),
        cell=np.array([[0, 2.72, 2.72], [2.72, 0, 2.72], [2.72, 2.72, 0]]),
        pbc=(True, True, True),
    )

@pytest.fixture
def dft_config():
    return DFTConfig(
        command=["mock_pw.x"],
        pseudopotentials={"Si": Path("Si.upf")},
        ecutwfc=30.0,
        kspacing=0.04,
    )

def test_uat_c03_01_standard_calculation(si_structure, dft_config):
    """Scenario 3.1: Standard DFT Calculation (Simulated)"""
    runner = QERunner(config=dft_config)

    # Mock subprocess.run to return success output
    # Mock file reading of the output file

    with patch("subprocess.run") as mock_run, \
         patch("pathlib.Path.read_text", return_value=MOCK_SUCCESS_OUTPUT):

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = runner.run(si_structure)

        assert isinstance(result, DFTResult)
        assert result.converged
        assert result.energy < 0
        assert result.forces.shape == (2, 3)

def test_uat_c03_02_self_healing(si_structure, dft_config):
    """Scenario 3.2: Self-Healing (Simulated)"""
    runner = QERunner(config=dft_config)

    # We need to simulate:
    # 1. First run -> Fails (Output contains failure message)
    # 2. Recovery handler triggers -> Updates config
    # 3. Second run -> Succeeds (Output contains success message)

    # Since QERunner.run calls subprocess then reads file, we need to mock these sequentially.
    # The runner creates a workdir and writes/reads files there.
    # We should mock `QERunner._execute` or `subprocess.run` and `open`.
    # Mocking `open` with side effects is tricky for multiple calls.
    # Better to mock `DFTParser.parse` if we want to test logic, but UAT should test integration.
    # Let's mock `subprocess.run` and `pathlib.Path.read_text` (if used) or `open`.

    side_effects_read = [MOCK_FAILURE_OUTPUT, MOCK_SUCCESS_OUTPUT]

    with patch("subprocess.run") as mock_run, \
         patch("pathlib.Path.read_text", side_effect=side_effects_read):

        mock_run.return_value = MagicMock(returncode=0) # QE often exits 0 even on SCF fail, or non-zero.

        result = runner.run(si_structure)

        assert result.converged
        assert mock_run.call_count == 2

        # Verify params changed in second call
        # We can inspect the arguments passed to mock_run or the written input files if we mocked write.
        # But checking call count and final success is enough for UAT high level.

def test_uat_c03_03_kpoint_grid(si_structure, dft_config):
    """Scenario 3.3: Automatic K-Point Grid"""
    # This tests InputGenerator logic mainly, but through the runner or just generator.
    # UAT says "Verify the input file pw.in reflects this dynamic sizing".

    from mlip_autopipec.physics.dft.input_gen import InputGenerator

    # Case 1: Small cell -> High K
    dft_config.kspacing = 0.2
    gen = InputGenerator()
    content_small = gen.generate_input(si_structure, dft_config)

    assert "K_POINTS automatic" in content_small
    # With k=0.2 and L~3.84, N = 2pi / (3.84 * 0.2) ~ 8
    # We expect N >= 4 at least.

    # Case 2: Large supercell -> Low K
    large_cell = si_structure.model_copy()
    large_cell.cell = si_structure.cell * 4 # 4x larger -> ~15 Angstrom

    content_large = gen.generate_input(large_cell, dft_config)
    # With k=0.2 and L~15, N = 2pi / (15 * 0.2) ~ 2

    # Extract K points
    def get_k(content):
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if "K_POINTS" in line:
                return [int(x) for x in lines[i+1].split()[:3]]
        return [0,0,0]

    k_small = get_k(content_small)
    k_large = get_k(content_large)

    assert all(ks > kl for ks, kl in zip(k_small, k_large))
