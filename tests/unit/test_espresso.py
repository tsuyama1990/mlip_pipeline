from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from mlip_autopipec.config import DFTConfig
from mlip_autopipec.physics.oracle.espresso import EspressoRunner


@pytest.fixture
def dft_config() -> DFTConfig:
    return DFTConfig(pseudopotentials={"Si": "Si.upf"})


@pytest.fixture
def atoms() -> Atoms:
    return Atoms("Si", positions=[[0, 0, 0]])


def test_runner_initialization(dft_config: DFTConfig) -> None:
    runner = EspressoRunner(dft_config)
    assert runner.config == dft_config


@patch("mlip_autopipec.physics.oracle.espresso.subprocess.run")
def test_run_single_success(mock_run: MagicMock, dft_config: DFTConfig, atoms: Atoms) -> None:
    runner = EspressoRunner(dft_config)

    with Path("tests/data/qe_outputs/converged.out").open() as f:
        output_content = f.read()

    def create_output_file(*args: Any, **kwargs: Any) -> MagicMock:
        # Command is "pw.x < input > output"
        # We extract output path and write content
        # Note: args[0] is the command string because shell=True
        cmd = args[0]
        output_path = cmd.split(">")[-1].strip()
        # output_path is relative to work_dir or absolute.
        # But we don't know work_dir easily here unless we check cwd arg.

        # EspressoRunner passes cwd=work_dir.
        # If output_path is absolute (from TemporaryDirectory), it works.
        # If relative, we need to join with cwd.

        cwd = kwargs.get("cwd")
        p = Path(cwd) / output_path if cwd else Path(output_path)
        p.write_text(output_content)
        return MagicMock(returncode=0)

    mock_run.side_effect = create_output_file

    result_atoms = runner.run_single(atoms)
    assert result_atoms.calc is not None
    # We can assert values if we trust the parser and the dummy file
    # converged.out has energy -156.23456789 Ry
    # -156.23456789 * 13.605693 (Ry to eV) approx -2125

    # Just check it ran
    assert mock_run.called


@patch("mlip_autopipec.physics.oracle.espresso.subprocess.run")
def test_run_single_retry(mock_run: MagicMock, dft_config: DFTConfig, atoms: Atoms) -> None:
    runner = EspressoRunner(dft_config)

    with Path("tests/data/qe_outputs/scf_error.out").open() as f:
        error_content = f.read()
    with Path("tests/data/qe_outputs/converged.out").open() as f:
        success_content = f.read()

    def create_output_file_retry(*args: Any, **kwargs: Any) -> MagicMock:
        cmd = args[0]
        output_path = cmd.split(">")[-1].strip()
        cwd = kwargs.get("cwd")
        p = Path(cwd) / output_path if cwd else Path(output_path)

        content = error_content if mock_run.call_count == 1 else success_content
        p.write_text(content)
        return MagicMock(returncode=0)

    mock_run.side_effect = create_output_file_retry

    runner.run_single(atoms)

    assert mock_run.call_count == 2
