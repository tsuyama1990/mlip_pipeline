"""Tests for pickle security."""

import gzip
import os
import pickle
import struct
from pathlib import Path

from loguru import logger

from pyacemaker.oracle.dataset import DatasetManager


class Malicious:
    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (os.system, ('echo "Attacked"',))

def test_pickle_security_rejection_logs(tmp_path: Path) -> None:
    """Test that loading malicious pickle content is rejected and logged."""
    dataset_path = tmp_path / "malicious.pckl.gzip"
    manager = DatasetManager()

    # Capture logs
    logs: list[str] = []
    logger.add(logs.append, format="{message}")

    # Create malicious payload
    obj_bytes = pickle.dumps(Malicious())
    size = len(obj_bytes)

    with gzip.open(dataset_path, "wb") as f:
        f.write(struct.pack(">Q", size))
        f.write(obj_bytes)

    # Run
    list(manager.load_iter(dataset_path, verify=False))

    # Verify rejection message in logs
    log_text = "\n".join(str(x) for x in logs)
    assert "Global 'posix.system' is forbidden" in log_text or "Global 'os.system' is forbidden" in log_text

def test_pickle_allowed_classes(tmp_path: Path) -> None:
    """Test that allowed classes (numpy, ase) are accepted."""
    import numpy as np
    from ase import Atoms

    dataset_path = tmp_path / "valid.pckl.gzip"
    manager = DatasetManager()

    # Valid atoms with numpy array
    atoms = Atoms("H")
    atoms.positions = np.array([[0.0, 0.0, 0.0]])

    manager.save_iter([atoms], dataset_path)

    loaded = list(manager.load_iter(dataset_path, verify=False))
    assert len(loaded) == 1
    assert isinstance(loaded[0], Atoms)
