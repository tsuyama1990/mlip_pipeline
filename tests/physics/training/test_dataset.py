from pathlib import Path
from unittest.mock import patch

import pytest
import numpy as np
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.training.dataset import DatasetManager


@pytest.fixture
def sample_structures():
    s1 = Structure(
        symbols=["Ti", "O"],
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        cell=np.eye(3),
        pbc=(True, True, True),
        properties={
            "energy": -10.0,
            "forces": np.zeros((2, 3)),
            "stress": np.zeros((3, 3)),
        },
    )
    return [s1]


def test_atoms_to_dataset_calls_pace_collect(sample_structures, tmp_path):
    manager = DatasetManager()
    output_path = tmp_path / "train.pckl.gzip"

    with patch("subprocess.run") as mock_run:
        result_path = manager.atoms_to_dataset(sample_structures, output_path)

        # Verify result path
        assert result_path == output_path

        # Verify subprocess call
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "pace_collect"
        assert args[-1] == str(output_path)
        assert "--extxyz" in args


def test_atoms_to_dataset_creates_intermediate_file(sample_structures, tmp_path):
    manager = DatasetManager()
    output_path = tmp_path / "train.pckl.gzip"

    with patch("subprocess.run") as mock_run:
        manager.atoms_to_dataset(sample_structures, output_path)

        # Extract the intermediate file path from the call
        args = mock_run.call_args[0][0]
        extxyz_path = Path(args[args.index("--extxyz") + 1])

        # Check that the file was created (even if mocked subprocess didn't consume it)
        # Note: In implementation, we expect the intermediate file to exist before subprocess call
        # and maybe deleted after (or not). Let's assume it exists during call.
        assert extxyz_path.exists()
        assert extxyz_path.suffix == ".extxyz"


def test_atoms_to_dataset_failure(sample_structures, tmp_path):
    manager = DatasetManager()
    output_path = tmp_path / "train.pckl.gzip"

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = RuntimeError("pace_collect failed")

        with pytest.raises(RuntimeError, match="pace_collect failed"):
            manager.atoms_to_dataset(sample_structures, output_path)
