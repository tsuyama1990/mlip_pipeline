from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mlip_autopipec.domain_models.calculation import DFTConfig, DFTResult
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.physics.dft.qe_runner import QERunner


@pytest.fixture
def uat_structure():
    return Structure(
        symbols=["Si", "Si"],
        positions=np.array([[0.0, 0.0, 0.0], [1.36, 1.36, 1.36]]),
        cell=np.array([[2.7, 0, 0], [0, 2.7, 0], [0, 0, 2.7]]),
        pbc=(True, True, True),
    )


@pytest.fixture
def uat_config():
    return DFTConfig(
        command="pw.x",
        pseudopotentials={"Si": Path("fake_si.upf")},
        ecutwfc=30.0,
        kspacing=0.05,
    )


def mock_subprocess_run_success(cmd, stdin, stdout, stderr, check, timeout):
    # Determine the job directory from stdout file path
    # stdout is a file object
    outfile_path = Path(stdout.name)

    # Write success output
    content = """
     Program PWSCF v.6.8 starts
     ...
     !    total energy              =     -150.00000000 Ry
     ...
     Forces acting on atoms (cartesian axes, Ry/au):
     atom    1 type  1   force =     0.01000000    0.00000000    0.00000000
     atom    2 type  1   force =    -0.01000000    0.00000000    0.00000000
     ...
     JOB DONE.
    """
    outfile_path.write_text(content)
    return MagicMock(returncode=0)

def mock_subprocess_run_fail_then_success(cmd, stdin, stdout, stderr, check, timeout):
    outfile_path = Path(stdout.name)
    # Check attempt number from filename?
    if "pw.out.1" in outfile_path.name:
        # Fail
        content = """
         ...
         convergence not achieved
        """
        outfile_path.write_text(content)
        # Should I raise CalledProcessError?
        # QE typically exits with 0 even if not converged, unless crash.
        # But 'convergence not achieved' is detected by Parser.
        # subprocess returncode=0 usually.
        return MagicMock(returncode=0)
    else:
        # Success
        content = """
         ...
         !    total energy              =     -150.00000000 Ry
         ...
         JOB DONE.
        """
        outfile_path.write_text(content)
        return MagicMock(returncode=0)


def test_uat_c03_01_standard_dft(uat_structure, uat_config, tmp_path):
    """
    Scenario 3.1: Standard DFT Calculation
    """
    # We need to mock DFTParser.parse because it relies on ASE to parse output.
    # ASE requires specific formatting.
    # To keep UAT robust without perfect mock output, I'll mock DFTParser.parse too,
    # OR verify the parser works on my mock output.
    # My mock output above is very skeletal. ASE won't parse it.
    # So I MUST mock DFTParser.parse to verify the runner logic, OR
    # I provide a real enough output.
    # Given complexity of QE output, mocking Parser is safer for UAT of the *Workflow*.

    success_result = DFTResult(
        job_id="uat_job_1",
        status=JobStatus.COMPLETED,
        work_dir=tmp_path,
        duration_seconds=5.0,
        log_content="Success",
        energy=-150.0 * 13.6, # approx Ry to eV
        forces=np.zeros((2, 3)),
        stress=np.zeros((3, 3))
    )

    with patch("mlip_autopipec.physics.dft.qe_runner.subprocess.run", side_effect=mock_subprocess_run_success) as mock_run, \
         patch("mlip_autopipec.physics.dft.qe_runner.DFTParser.parse", return_value=success_result) as mock_parse:

        runner = QERunner(uat_config, work_dir=tmp_path)
        result = runner.run(uat_structure, job_id="uat_job_1")

        assert result.status == JobStatus.COMPLETED
        assert result.energy < 0
        assert mock_run.call_count == 1

        # Verify inputs generated
        input_file = tmp_path / "uat_job_1" / "pw.in.1"
        assert input_file.exists()
        content = input_file.read_text()
        assert "K_POINTS automatic" in content


def test_uat_c03_02_self_healing(uat_structure, uat_config, tmp_path):
    """
    Scenario 3.2: Self-Healing
    """
    from mlip_autopipec.domain_models.calculation import SCFError

    success_result = DFTResult(
        job_id="uat_job_2",
        status=JobStatus.COMPLETED,
        work_dir=tmp_path,
        duration_seconds=5.0,
        log_content="Success",
        energy=-150.0,
        forces=np.zeros((2, 3))
    )

    # Mock Parser to raise SCFError first, then return Success
    with patch("mlip_autopipec.physics.dft.qe_runner.subprocess.run", side_effect=mock_subprocess_run_fail_then_success) as mock_run, \
         patch("mlip_autopipec.physics.dft.qe_runner.DFTParser.parse", side_effect=[SCFError("SCF Failed"), success_result]) as mock_parse:

        runner = QERunner(uat_config, work_dir=tmp_path)
        result = runner.run(uat_structure, job_id="uat_job_2")

        assert result.status == JobStatus.COMPLETED
        assert mock_run.call_count == 2
        assert mock_parse.call_count == 2

        # Verify attempt 2 input has different params
        input_file_2 = tmp_path / "uat_job_2" / "pw.in.2"
        assert input_file_2.exists()
        content_2 = input_file_2.read_text()
        # Expect mixing_beta = 0.3 (first strategy)
        # Note: input generator flattening puts mixing_beta in params
        import re
        assert re.search(r"mixing_beta\s*=\s*0\.3", content_2, re.IGNORECASE)

def test_uat_c03_03_kpoint_grid(uat_structure, uat_config, tmp_path):
    """
    Scenario 3.3: Automatic K-Point Grid
    """
    # Verify input generation logic specifically for k-points
    runner = QERunner(uat_config, work_dir=tmp_path)

    # Run creates input file.
    # We mock execution to stop immediately.

    with patch("mlip_autopipec.physics.dft.qe_runner.subprocess.run") as mock_run, \
         patch("mlip_autopipec.physics.dft.qe_runner.DFTParser.parse") as mock_parse:

        mock_run.side_effect = Exception("Stop execution") # Quick abort

        try:
            runner.run(uat_structure, job_id="uat_job_3")
        except Exception:
            pass

        input_file = tmp_path / "uat_job_3" / "pw.in.1"
        assert input_file.exists()
        content = input_file.read_text()

        # Structure is 2.7 Angstrom. kspacing 0.05.
        # Expect dense grid.
        # Check "K_POINTS automatic" followed by numbers
        # regex search
        import re
        match = re.search(r"K_POINTS automatic\s*\n\s*(\d+)\s+(\d+)\s+(\d+)", content)
        if match:
            kpts = [int(g) for g in match.groups()]
            assert all(k >= 20 for k in kpts) # 2.7 cell -> >40 kpoints
        else:
            pytest.fail("K_POINTS automatic not found or malformed")
