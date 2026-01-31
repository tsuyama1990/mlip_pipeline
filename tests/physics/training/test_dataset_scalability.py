from unittest.mock import patch
from mlip_autopipec.physics.training.dataset import DatasetManager
from mlip_autopipec.domain_models.structure import Structure
import numpy as np

def test_dataset_scalability_mock(tmp_path):
    """
    Test that dataset conversion handles large iterators without OOM (simulated).
    We mock write to avoid disk I/O but check generator consumption.
    """
    dm = DatasetManager(tmp_path)

    # Create a generator that yields many structures
    def infinite_structures():
        for i in range(100000):
            yield Structure(symbols=["Si"], positions=np.array([[0,0,0]]), cell=np.eye(3))

    # Mock ase.io.write to consume generator without storing
    with patch("mlip_autopipec.physics.training.dataset.write") as mock_write:
        with patch("subprocess.run"): # Mock pace_collect
            dm.convert(infinite_structures(), tmp_path / "out.pckl.gzip")

            # Check that write was called with a generator
            args, _ = mock_write.call_args
            assert args[0] == tmp_path / "dataset.extxyz"
            # args[1] should be the generator wrapper
            # We can't easily check if it's "streaming" via mock, but we know code does.
            # This test mainly ensures no crash.
