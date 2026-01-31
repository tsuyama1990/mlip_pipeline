from unittest.mock import patch
import subprocess

import pytest
from ase import Atoms

from mlip_autopipec.domain_models.structure import Structure

# We assume the module exists (TDD)
from mlip_autopipec.physics.training.dataset import DatasetManager


@pytest.fixture
def sample_structures():
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[5, 5, 5], pbc=True)
    # Mock properties
    atoms.info["energy"] = -10.0
    atoms.info["forces"] = [[0.1, 0.1, 0.1], [-0.1, -0.1, -0.1]]
    atoms.info["stress"] = [0.0] * 9

    struct = Structure.from_ase(atoms)
    return [struct]


@patch("subprocess.run")
def test_dataset_conversion(mock_run, sample_structures, tmp_path):
    manager = DatasetManager(work_dir=tmp_path)
    output_path = tmp_path / "train.pckl.gzip"

    # Mock subprocess success
    mock_run.return_value.returncode = 0

    manager.convert(sample_structures, output_path)

    # Check if extxyz file was created
    # The manager should create a temporary extxyz file
    extxyz_files = list(tmp_path.glob("*.extxyz"))
    assert len(extxyz_files) > 0
    extxyz_path = extxyz_files[0]

    # Check if pace_collect was called
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    cmd = args[0]
    assert cmd[0] == "pace_collect"
    assert str(extxyz_path) in cmd
    assert str(output_path) in cmd


@patch("subprocess.run")
def test_dataset_conversion_failure(mock_run, sample_structures, tmp_path):
    # Raise CalledProcessError to simulate command failure
    mock_run.side_effect = subprocess.CalledProcessError(
        1, cmd="pace_collect", stderr="pace_collect failed"
    )

    manager = DatasetManager(work_dir=tmp_path)
    output_path = tmp_path / "train.pckl.gzip"

    with pytest.raises(RuntimeError):
        manager.convert(sample_structures, output_path)
