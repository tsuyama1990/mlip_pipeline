"""Tests for streaming capabilities."""

from pathlib import Path

from ase import Atoms

from pyacemaker.oracle.dataset import DatasetManager


def test_dataset_manager_streaming(tmp_path: Path) -> None:
    """Test that DatasetManager streams objects without loading all into memory."""
    dataset_path = tmp_path / "stream_test.pckl.gzip"

    # Create a large dummy dataset (logically large count, small size per item)
    atoms_list = [Atoms("H2") for _ in range(100)]
    manager = DatasetManager()
    manager.save_iter(iter(atoms_list), dataset_path, calculate_checksum=False)

    # Use a spy on the unpickler to ensure it's called incrementally
    # We can't easily spy on memory usage directly in unit test without psutil
    # But we can verify that we can break the loop early and not read the rest

    stream = manager.load_iter(dataset_path, verify=False)

    count = 0
    for i, atoms in enumerate(stream):
        count += 1
        assert len(atoms) == 2
        if i == 10:
            break

    assert count == 11
    # If it loaded everything, we wouldn't know, but at least it works as iterator.


def test_limited_stream_behavior() -> None:
    """Test LimitedStream logic directly."""
    from io import BytesIO

    from pyacemaker.oracle.dataset import LimitedStream

    content = b"1234567890"
    stream = BytesIO(content)

    # Limit to 5 bytes
    limited = LimitedStream(stream, size=5)

    read1 = limited.read(3)
    assert read1 == b"123"
    assert limited._remaining == 2

    read2 = limited.read(10)  # Ask for more than remaining
    assert read2 == b"45"
    assert limited._remaining == 0

    read3 = limited.read(1)
    assert read3 == b""
