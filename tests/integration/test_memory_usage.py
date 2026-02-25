"""Memory usage profiling for streaming operations."""

import tracemalloc
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from ase import Atoms

from pyacemaker.core.dataset import DatasetSplitter
from pyacemaker.oracle.dataset import DatasetManager


# Utility to measure memory
def measure_memory_peak(func: Callable[..., Any], *args: Any, **kwargs: Any) -> float:
    """Run function and return peak memory usage in MB."""
    tracemalloc.start()
    func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def test_dataset_manager_streaming_memory(tmp_path: Path) -> None:
    """Test that DatasetManager.save_iter and load_iter are memory efficient."""
    dataset_path = tmp_path / "large_dataset.pckl.gzip"

    # Generate a large dataset
    # 50,000 items to verify streaming
    n_items = 50000

    def data_gen() -> Iterator[Atoms]:
        for _ in range(n_items):
            yield Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])

    manager = DatasetManager()

    # Measure Save Memory
    # Should be constant relative to n_items
    peak_mb = measure_memory_peak(manager.save_iter, data_gen(), dataset_path)

    # Calculate expected memory usage
    # Buffer size (10MB default) + Python overhead
    # Adjusted threshold to be realistic but still verify streaming
    # 50k atoms objects if loaded would be > 200MB easily.
    # We expect < 100MB to account for buffer + overhead.
    expected_max_mb = 100.0

    assert peak_mb < expected_max_mb

    # Measure Load Memory
    def consume_load() -> None:
        from collections import deque

        # Consume without storing
        deque(manager.load_iter(dataset_path, verify=False), maxlen=0)

    peak_mb_load = measure_memory_peak(consume_load)
    assert peak_mb_load < expected_max_mb


def test_dataset_splitter_memory(tmp_path: Path) -> None:
    """Test that DatasetSplitter streams efficiently."""
    dataset_path = tmp_path / "dataset.pckl.gzip"
    validation_path = tmp_path / "validation.pckl.gzip"

    manager = DatasetManager()

    # create dataset using generator
    n_items = 50000

    def data_gen() -> Iterator[Atoms]:
        for _ in range(n_items):
            yield Atoms("Fe")

    manager.save_iter(data_gen(), dataset_path)

    splitter = DatasetSplitter(
        dataset_path,
        validation_path,
        manager,
        validation_split=0.2,
        max_validation_size=1000,
        buffer_size=100,  # Flush every 100 items
    )

    # We patch load_iter to ensure verify=False to avoid reading whole file for checksum first
    from unittest.mock import patch

    def consume_split() -> None:
        with patch.object(
            manager,
            "load_iter",
            side_effect=lambda p, **kwargs: DatasetManager.load_iter(
                manager, p, verify=False, **kwargs
            ),
        ):
            # Consume generator
            # This verifies that splitter logic processes items in stream
            for _ in splitter.train_stream():
                pass

    peak_mb = measure_memory_peak(consume_split)

    # Should not load all items
    expected_max_mb = 100.0
    assert peak_mb < expected_max_mb
