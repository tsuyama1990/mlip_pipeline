import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import subprocess

from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.training.dataset import DatasetManager

@pytest.fixture
def sample_structures():
    s1 = Structure(
        symbols=["Ti", "O"],
        positions=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        cell=np.eye(3) * 10.0,
        pbc=(True, True, True),
        properties={"energy": -10.0, "forces": np.zeros((2, 3)), "stress": np.zeros((3, 3))}
    )
    s2 = Structure(
        symbols=["Ti", "O"],
        positions=np.array([[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]]),
        cell=np.eye(3) * 10.0,
        pbc=(True, True, True),
        properties={"energy": -10.5, "forces": np.zeros((2, 3)), "stress": np.zeros((3, 3))}
    )
    return [s1, s2]

def test_dataset_manager_conversion(sample_structures, tmp_path):
    """Test that atoms are written to extxyz and pace_collect is called."""
    manager = DatasetManager(work_dir=tmp_path)

    with patch("subprocess.run") as mock_run:
        # Side effect to create the output file
        def side_effect(*args, **kwargs):
             (tmp_path / "train_set.pckl.gzip").touch()
             return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        output_path = manager.create_dataset(sample_structures, "train_set")

        assert output_path == tmp_path / "train_set.pckl.gzip"

        # Check that extxyz file was created
        extxyz_path = tmp_path / "train_set.extxyz"
        assert extxyz_path.exists()

        # Verify subprocess call
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "pace_collect"
        cmd_str = " ".join(str(a) for a in args)
        assert str(extxyz_path) in cmd_str
        assert str(output_path) in cmd_str

def test_dataset_manager_failure_process(sample_structures, tmp_path):
    """Test handling of pace_collect failure."""
    manager = DatasetManager(work_dir=tmp_path)

    with patch("subprocess.run") as mock_run:
        # subprocess.CalledProcessError requires returncode, cmd
        mock_run.side_effect = subprocess.CalledProcessError(1, "pace_collect", stderr="Error")

        with pytest.raises(RuntimeError, match="Dataset conversion failed"):
            manager.create_dataset(sample_structures, "train_set")

def test_dataset_manager_failure_not_found(sample_structures, tmp_path):
    """Test handling of missing pace_collect executable."""
    manager = DatasetManager(work_dir=tmp_path)

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError("pace_collect")

        with pytest.raises(RuntimeError, match="pace_collect executable not found"):
            manager.create_dataset(sample_structures, "train_set")

def test_dataset_manager_no_output(sample_structures, tmp_path):
    """Test case where pace_collect runs but output file is not created."""
    manager = DatasetManager(work_dir=tmp_path)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        # No side effect to create file

        with pytest.raises(RuntimeError, match="Dataset file was not created"):
            manager.create_dataset(sample_structures, "train_set")
