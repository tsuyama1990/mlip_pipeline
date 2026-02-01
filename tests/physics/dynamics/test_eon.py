import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from mlip_autopipec.domain_models.dynamics import EonConfig, EonResult
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.domain_models.config import PotentialConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.dynamics.eon import EonWrapper
import subprocess
import ase

@pytest.fixture
def mock_structure():
    atoms = ase.Atoms("Pt4", positions=[[0,0,0], [2,0,0], [0,2,0], [0,0,2]], cell=[10,10,10], pbc=True)
    return Structure.from_ase(atoms)

@pytest.fixture
def eon_config():
    return EonConfig(command="eon", timeout=1)

@pytest.fixture
def potential_config():
    return PotentialConfig(elements=["Pt"])

def test_eon_wrapper_init(eon_config, potential_config):
    wrapper = EonWrapper(eon_config, potential_config, Path("/tmp/work"))
    assert wrapper.base_work_dir == Path("/tmp/work")

@patch("subprocess.Popen")
@patch("shutil.which")
def test_eon_run_success(mock_which, mock_popen, eon_config, potential_config, mock_structure, tmp_path):
    mock_which.return_value = "/usr/bin/eon"

    # Mock subprocess
    process_mock = MagicMock()
    process_mock.stdout.fileno.return_value = 1
    process_mock.stderr.fileno.return_value = 2
    process_mock.stdout.read.side_effect = [b"EON output", b""]
    process_mock.stderr.read.return_value = b""
    process_mock.poll.side_effect = [None, 0]
    process_mock.returncode = 0
    mock_popen.return_value = process_mock

    wrapper = EonWrapper(eon_config, potential_config, tmp_path)

    # We mock _parse_output to return a specific result to verifying 'run' flow
    # independently of parsing logic
    with patch.object(EonWrapper, "_parse_output") as mock_parse:
        mock_parse.return_value = EonResult(
            job_id="test_id",
            status=JobStatus.COMPLETED,
            work_dir=tmp_path,
            duration_seconds=1.0,
            log_content="done",
            final_structure=mock_structure,
            max_gamma=4.5
        )

        result = wrapper.run(mock_structure, Path("potential.yace"))

        assert result.status == JobStatus.COMPLETED
        assert result.max_gamma == 4.5

        # Check inputs generated
        subdirs = list(tmp_path.glob("akmc_*"))
        assert len(subdirs) == 1
        job_dir = subdirs[0]

        assert (job_dir / "client" / "config.ini").exists()
        assert (job_dir / "pos.con").exists()

@patch("subprocess.Popen")
def test_eon_run_timeout(mock_popen, eon_config, potential_config, mock_structure, tmp_path):
    # Mock timeout
    process_mock = MagicMock()
    process_mock.wait.side_effect = subprocess.TimeoutExpired(cmd="eon", timeout=1)
    mock_popen.return_value = process_mock

    wrapper = EonWrapper(eon_config, potential_config, tmp_path)

    result = wrapper.run(mock_structure, Path("potential.yace"))

    # Should catch exception and return Result with error or check logs
    # EonWrapper catches Exception and returns FAILED
    # Wait, code catches subprocess.TimeoutExpired but then continues to _parse_output?
    # No:
    # try:
    #     process.wait(timeout=self.config.timeout)
    # except subprocess.TimeoutExpired:
    #     process.terminate()
    #     logger.warning("EON timed out")
    #
    # return self._parse_output(work_dir)

    # Should catch exception and return Result with TIMEOUT status
    assert result.status == JobStatus.TIMEOUT
    assert result.log_content == "Timeout Expired"

def test_parse_output_high_gamma(eon_config, potential_config, tmp_path):
    wrapper = EonWrapper(eon_config, potential_config, tmp_path)
    work_dir = tmp_path / "job_dir"
    work_dir.mkdir()

    # Write stderr with High Gamma message
    (work_dir / eon_config.stderr_file).write_text("Some logs...\nHigh Gamma Detected: 7.5 > 5.0\nExiting.")

    # Write pos.con
    atoms = ase.Atoms("H2", positions=[[0,0,0], [0,0,1]])
    ase.io.write(work_dir / "pos.con", atoms, format="eon") # type: ignore

    result = wrapper._parse_output(work_dir)

    assert result.max_gamma == 7.5
    assert result.final_structure is not None
    assert len(result.final_structure.symbols) == 2

def test_parse_output_no_gamma(eon_config, potential_config, tmp_path):
    wrapper = EonWrapper(eon_config, potential_config, tmp_path)
    work_dir = tmp_path / "job_dir"
    work_dir.mkdir()

    # Normal stderr
    (work_dir / eon_config.stderr_file).write_text("Normal run.")

    result = wrapper._parse_output(work_dir)

    assert result.max_gamma == 0.0
