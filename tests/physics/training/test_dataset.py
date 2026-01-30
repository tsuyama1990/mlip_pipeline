from unittest.mock import patch

import pytest
from ase.atoms import Atoms
import numpy as np

from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.training.dataset import DatasetManager


@pytest.fixture
def sample_structures():
    s1 = Structure.from_ase(Atoms("Si", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True))
    # Add dummy properties
    s1.properties["energy"] = -10.0
    s1.properties["forces"] = np.array([[0.1, 0.2, 0.3]])
    s1.properties["stress"] = np.zeros((3, 3))

    s2 = Structure.from_ase(Atoms("Si", positions=[[2.5, 2.5, 2.5]], cell=[5, 5, 5], pbc=True))
    s2.properties["energy"] = -10.1
    s2.properties["forces"] = np.array([[0.0, 0.0, 0.0]])
    s2.properties["stress"] = np.eye(3) * 0.1

    return [s1, s2]


def test_convert_calls_pace_collect(sample_structures, tmp_path):
    manager = DatasetManager()
    output_path = tmp_path / "train.pckl.gzip"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0

        result_path = manager.convert(sample_structures, output_path)

        assert result_path == output_path

        # Verify extxyz was created
        extxyz_path = tmp_path / "temp_dataset.extxyz"
        assert extxyz_path.exists()

        # Verify pace_collect was called
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "pace_collect" in args
        assert str(extxyz_path) in args
        assert str(output_path) in args


def test_convert_handles_failure(sample_structures, tmp_path):
    manager = DatasetManager()
    output_path = tmp_path / "train.pckl.gzip"

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError("pace_collect not found")

        with pytest.raises(RuntimeError):
            manager.convert(sample_structures, output_path)
