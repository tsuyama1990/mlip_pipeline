"""
Unit tests for the refactored DFTFactory and its dependencies.
"""
import subprocess
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.espresso import EspressoProfile

from mlip_autopipec.config.models import DFTInputParameters, DFTJob
from mlip_autopipec.exceptions import DFTCalculationError
from mlip_autopipec.modules.dft import (
    DFTJobFactory,
    DFTRunner,
    QEInputGenerator,
    QEOutputParser,
    QEProcessRunner,
    QERetryHandler,
)


@pytest.fixture
def mock_input_generator():
    return MagicMock(spec=QEInputGenerator)


@pytest.fixture
def mock_process_runner():
    return MagicMock(spec=QEProcessRunner)


@pytest.fixture
def mock_output_parser():
    return MagicMock(spec=QEOutputParser)


@pytest.fixture
def mock_retry_handler():
    return MagicMock(spec=QERetryHandler)


@pytest.fixture
def dft_runner(
    mock_input_generator,
    mock_process_runner,
    mock_output_parser,
    mock_retry_handler,
):
    """Fixture to create a DFTRunner with mocked dependencies."""
    return DFTRunner(
        input_generator=mock_input_generator,
        process_runner=mock_process_runner,
        output_parser=mock_output_parser,
        retry_handler=mock_retry_handler,
    )


def test_dft_runner_run_successful(
    dft_runner,
    mock_input_generator,
    mock_process_runner,
    mock_output_parser,
):
    """Test a straightforward, successful run."""
    atoms = bulk("Si")
    params = MagicMock(spec=DFTInputParameters)
    job = DFTJob(atoms=atoms, params=params)
    mock_output_parser.parse.return_value = MagicMock(energy=-100.0)

    result = dft_runner.run(job)

    mock_input_generator.prepare_input_files.assert_called_with(ANY, atoms, params)
    mock_process_runner.execute.assert_called_once()
    mock_output_parser.parse.assert_called_with(ANY, job.job_id)
    assert result.energy == -100.0


def test_dft_runner_retry_and_succeed(
    dft_runner,
    mock_input_generator,
    mock_process_runner,
    mock_output_parser,
    mock_retry_handler,
):
    """Test the retry mechanism where a run fails once, then succeeds."""
    atoms = bulk("Si")
    error = subprocess.CalledProcessError(1, "pw.x")
    error.stdout = "convergence NOT achieved"
    error.stderr = ""
    mock_process_runner.execute.side_effect = [error, MagicMock()]
    mock_retry_handler.handle_convergence_error.return_value = {"mixing_beta": 0.35}
    mock_output_parser.parse.return_value = MagicMock(energy=-150.0)
    params = MagicMock(spec=DFTInputParameters)
    params.model_dump.return_value = {
            "pseudopotentials": {"Si": "Si.upf"},
            "cutoffs": {"wavefunction": 60, "density": 240},
            "k_points": (3, 3, 3),
            "mixing_beta": 0.7,
        }
    job = DFTJob(atoms=atoms, params=params)

    result = dft_runner.run(job)

    assert mock_process_runner.execute.call_count == 2
    mock_retry_handler.handle_convergence_error.assert_called_with(
        "convergence NOT achieved\\n",
        {
            "pseudopotentials": {"Si": "Si.upf"},
            "cutoffs": {"wavefunction": 60, "density": 240},
            "k_points": (3, 3, 3),
            "mixing_beta": 0.35,
        },
    )
    assert result.energy == -150.0


def test_dft_runner_fails_after_max_retries(
    dft_runner,
    mock_input_generator,
    mock_process_runner,
    mock_retry_handler,
):
    """Test that the factory gives up after the maximum number of retries."""
    dft_runner.max_retries = 3
    atoms = bulk("Si")
    error = subprocess.CalledProcessError(1, "pw.x")
    error.stdout = "convergence NOT achieved"
    error.stderr = ""
    mock_process_runner.execute.side_effect = [error, error, error]
    mock_retry_handler.handle_convergence_error.return_value = {"mixing_beta": 0.35}
    params = MagicMock(spec=DFTInputParameters)
    params.model_dump.return_value = {
            "pseudopotentials": {"Si": "Si.upf"},
            "cutoffs": {"wavefunction": 60, "density": 240},
            "k_points": (3, 3, 3),
            "mixing_beta": 0.7,
        }
    job = DFTJob(atoms=atoms, params=params)

    with pytest.raises(DFTCalculationError):
        dft_runner.run(job)

    assert mock_process_runner.execute.call_count == 3
    assert mock_retry_handler.handle_convergence_error.call_count == 3


def test_handle_convergence_error():
    """Unit test for the convergence error handling logic."""
    retry_handler = QERetryHandler()
    params = {"mixing_beta": 0.7}
    log = "convergence NOT achieved"
    new_params = retry_handler.handle_convergence_error(log, params)
    assert new_params is not None
    assert new_params["mixing_beta"] == 0.35

    params = {"diagonalization": "david"}
    log = "Cholesky"
    new_params = retry_handler.handle_convergence_error(log, params)
    assert new_params is not None
    assert new_params["diagonalization"] == "cg"

    params = {}
    log = "some other error"
    new_params = retry_handler.handle_convergence_error(log, params)
    assert new_params is None


def test_process_runner_logs_on_error(caplog, tmp_path):
    """Verify that the QEProcessRunner logs stdout/stderr on failure."""
    mock_profile = MagicMock(spec=EspressoProfile)
    mock_profile.get_command.return_value = ["non_existent_command"]
    runner = QEProcessRunner(profile=mock_profile)

    input_file = tmp_path / "test.in"
    output_file = tmp_path / "test.out"
    input_file.touch()

    with pytest.raises(FileNotFoundError):
        runner.execute(input_file, output_file)
    assert "QE executable not found" in caplog.text


import uuid


def test_output_parser_handles_valid_output(tmp_path):
    """Test that the QEOutputParser correctly parses a valid output file."""
    output_file = tmp_path / "espresso.pwo"
    output_file.touch()  # Create the file

    mock_atoms = bulk("Si")
    mock_atoms.set_cell([5.43, 5.43, 5.43, 90, 90, 90])
    mock_atoms.set_pbc(True)
    calc = MagicMock()
    calc.get_potential_energy.return_value = -136.057
    calc.get_forces.return_value = [[0.0, 0.0, 0.0]]
    calc.get_stress.return_value = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mock_atoms.calc = calc
    mock_reader = MagicMock(return_value=mock_atoms)
    parser = QEOutputParser(reader=mock_reader)

    result = parser.parse(output_file, uuid.uuid4())

    assert result.energy == -136.057
    assert len(result.forces) == 1
    mock_reader.assert_called_once_with(output_file, format="espresso-out")


def test_dft_job_factory_creates_valid_job(tmp_path):
    """Test the heuristic parameter generation in DFTJobFactory."""
    sssp_path = tmp_path / "sssp.json"
    sssp_path.write_text('{"Si": {"cutoff_wfc": 60, "cutoff_rho": 240, "filename": "Si.upf"}}')
    with patch(
        "mlip_autopipec.modules.dft.SSSP_DATA_PATH",
        sssp_path,
    ):
        factory = DFTJobFactory()
        atoms = bulk("Si")
        job = factory.create_job(atoms)
        assert isinstance(job, DFTJob)
        assert job.params.cutoffs.wavefunction == 60
        assert job.params.cutoffs.density == 240
        assert job.params.pseudopotentials is not None
        assert job.params.pseudopotentials.root["Si"] == "Si.upf"

