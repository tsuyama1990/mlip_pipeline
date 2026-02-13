from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.trainer.dataset_manager import DatasetManager


@pytest.fixture
def sample_structures() -> list[Structure]:
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    s1 = Structure(atoms=atoms, provenance="test", label_status="labeled", energy=-10.0, forces=[[0,0,0]]*3, stress=[0]*6)
    s2 = Structure(atoms=atoms, provenance="test", label_status="labeled", energy=-11.0, forces=[[0,0,0]]*3, stress=[0]*6)
    return [s1, s2]

@patch("subprocess.run")
def test_create_dataset(mock_run: MagicMock, sample_structures: list[Structure], tmp_path: Path) -> None:
    manager = DatasetManager(work_dir=tmp_path)
    output_path = tmp_path / "dataset.pckl.gzip"

    # We expect create_dataset to first write structures to a temporary file (e.g. extxyz)
    # then call pace_collect on that file.

    # Mock return code 0
    mock_run.return_value.returncode = 0
    # Create the output file since manager checks for its existence
    output_path.touch()

    manager.create_dataset(sample_structures, output_path)

    assert mock_run.called
    args = mock_run.call_args[0][0]
    assert args[0] == "pace_collect"
    assert output_path.name in args
    # verify input file is passed (it will be a temp file, so we check general structure)
    assert any(str(arg).endswith(".extxyz") for arg in args)

@patch("subprocess.run")
def test_select_active_set(mock_run: MagicMock, tmp_path: Path) -> None:
    manager = DatasetManager(work_dir=tmp_path)
    dataset_path = tmp_path / "dataset.pckl.gzip"
    # Mock return code 0
    mock_run.return_value.returncode = 0

    # Create the output file since manager checks for its existence
    output_path = tmp_path / f"active_set_{10}.pckl.gzip"
    output_path.touch()

    manager.select_active_set(dataset_path, count=10)

    assert mock_run.called
    args = mock_run.call_args[0][0]
    assert args[0] == "pace_activeset"
    assert dataset_path.name in args
    assert "--max_size" in args
    assert "10" in args

@patch("subprocess.run")
def test_create_dataset_failure(mock_run: MagicMock, sample_structures: list[Structure], tmp_path: Path) -> None:
    manager = DatasetManager(work_dir=tmp_path)
    output_path = tmp_path / "dataset.pckl.gzip"

    # Mock return code 1
    mock_run.return_value.returncode = 1
    mock_run.return_value.stderr = "Error occurred"

    with pytest.raises(RuntimeError, match="pace_collect failed"):
        manager.create_dataset(sample_structures, output_path)

@patch("subprocess.run")
def test_create_dataset_missing_output(mock_run: MagicMock, sample_structures: list[Structure], tmp_path: Path) -> None:
    manager = DatasetManager(work_dir=tmp_path)
    output_path = tmp_path / "dataset.pckl.gzip"

    # Mock return code 0 but do NOT create file
    mock_run.return_value.returncode = 0
    if output_path.exists():
        output_path.unlink()

    with pytest.raises(FileNotFoundError, match="pace_collect did not produce output"):
        manager.create_dataset(sample_structures, output_path)

@patch("subprocess.run")
def test_select_active_set_failure(mock_run: MagicMock, tmp_path: Path) -> None:
    manager = DatasetManager(work_dir=tmp_path)
    dataset_path = tmp_path / "dataset.pckl.gzip"

    mock_run.return_value.returncode = 1
    mock_run.return_value.stderr = "Error occurred"

    with pytest.raises(RuntimeError, match="pace_activeset failed"):
        manager.select_active_set(dataset_path, count=10)

@patch("subprocess.run")
def test_select_active_set_missing_output(mock_run: MagicMock, tmp_path: Path) -> None:
    manager = DatasetManager(work_dir=tmp_path)
    dataset_path = tmp_path / "dataset.pckl.gzip"
    output_path = tmp_path / f"active_set_{10}.pckl.gzip"

    mock_run.return_value.returncode = 0
    if output_path.exists():
        output_path.unlink()

    with pytest.raises(FileNotFoundError, match="pace_activeset did not produce output"):
        manager.select_active_set(dataset_path, count=10)
