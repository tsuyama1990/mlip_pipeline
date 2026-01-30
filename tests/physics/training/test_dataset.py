from unittest.mock import patch

import pytest
from ase.build import bulk

from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.training.dataset import DatasetManager


@pytest.fixture
def sample_structures():
    atoms1 = bulk("Ti", "hcp", a=2.95)
    atoms1.info["energy"] = -100.0
    atoms1.info["forces"] = [[0, 0, 0]] * len(atoms1)
    atoms1.info["stress"] = [0] * 6

    atoms2 = bulk("O", "fcc", a=4.0)
    atoms2.info["energy"] = -200.0
    atoms2.info["forces"] = [[0, 0, 0]] * len(atoms2)
    atoms2.info["stress"] = [0] * 6

    return [Structure.from_ase(atoms1), Structure.from_ase(atoms2)]


def test_dataset_conversion(sample_structures, tmp_path):
    manager = DatasetManager(work_dir=tmp_path)
    output_path = tmp_path / "train.pckl.gzip"

    # Mock subprocess.run to avoid actual pace_collect execution
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0

        # We also need to mock the existence of the output file because the manager might check it
        # But we can just assume the manager trusts subprocess if returncode is 0.
        # However, usually we might want to verify the intermediate extxyz file was created.

        result_path = manager.convert(sample_structures, output_path)

        assert result_path == output_path

        # Check if subprocess was called
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "pace_collect" in args[0] or "pace_collect" in args
        assert str(output_path) in args

        # Verify intermediate file creation
        # The manager should create an extxyz file before calling pace_collect
        # We can check if any .extxyz file exists in tmp_path
        extxyz_files = list(tmp_path.glob("*.extxyz"))
        assert len(extxyz_files) > 0

        # Check content of extxyz (basic check)
        content = extxyz_files[0].read_text()
        assert "Lattice=" in content
        assert "energy=" in content
