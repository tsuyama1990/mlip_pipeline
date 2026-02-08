from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from mlip_autopipec.core.dataset import Dataset
from mlip_autopipec.domain_models.structure import Structure


def test_dataset_streaming_behavior(tmp_path: Path) -> None:
    """Verify that dataset iteration is lazy and does not load all lines at once."""
    dataset_path = tmp_path / "stream_test.jsonl"
    dataset_path.touch()  # Ensure it exists
    dataset_path.with_suffix(".meta.json").write_text('{"count": 0}')

    # Create a dummy structure for serialization
    import numpy as np

    s = Structure(
        positions=np.zeros((1, 3)),
        atomic_numbers=np.array([1]),
        cell=np.eye(3),
        pbc=np.array([True, True, True]),
        energy=-1.5,
        forces=np.zeros((1, 3)),
        stress=np.zeros((3, 3)),
    )
    line = s.model_dump_json() + "\n"

    ds = Dataset(dataset_path)

    # We patch Path.open to return a mock file that we can spy on
    # We configure the mock file to yield lines via generator to prove streaming
    mock_file_handle = MagicMock()
    mock_file_handle.__enter__.return_value = mock_file_handle

    # Use a generator to enforce lazy iteration
    def line_generator() -> Iterator[str]:
        for _ in range(3):
            yield line

    mock_file_handle.__iter__.return_value = line_generator()

    with patch("pathlib.Path.open", return_value=mock_file_handle):
        # iter(ds) checks self.path.exists(). Since we created it, it exists.
        # Then it calls self.path.open("r")

        iterator = iter(ds)

        # Verify we have an iterator
        import collections.abc

        assert isinstance(iterator, collections.abc.Iterator)
        assert not isinstance(iterator, list)

        # Consume one
        # If open wasn't called, this would fail or read from empty real file (if patch didn't work)
        item = next(iterator)
        assert isinstance(item, Structure)
        assert np.allclose(item.positions, s.positions)

        # Verify that we iterated over the file handle exactly once
        mock_file_handle.__iter__.assert_called_once()


def test_dataset_append_buffering(tmp_path: Path) -> None:
    """Verify that append uses buffering."""
    dataset_path = tmp_path / "buffer_test.jsonl"
    meta_path = dataset_path.with_suffix(".meta.json")

    dataset_path.touch()
    meta_path.write_text('{"count": 0}')

    ds = Dataset(dataset_path)

    import numpy as np

    s = Structure(
        positions=np.zeros((1, 3)),
        atomic_numbers=np.array([1]),
        cell=np.eye(3),
        pbc=np.array([True, True, True]),
        energy=-1.5,
        forces=np.zeros((1, 3)),
        stress=np.zeros((3, 3)),
    )

    structures = [s for _ in range(5)]

    # Save original open to use for passthrough
    original_open = Path.open

    # Create the mock handle for the dataset append
    append_handle = MagicMock()
    append_handle.__enter__.return_value = append_handle
    append_handle.__exit__.return_value = None

    def side_effect(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Check if we are opening the dataset in append mode
        if str(self) == str(dataset_path) and (
            (len(args) > 0 and args[0] == "a") or kwargs.get("mode") == "a"
        ):
            return append_handle
        # Otherwise use real file ops (e.g. for metadata reading/writing)
        return original_open(self, *args, **kwargs)

    with patch("pathlib.Path.open", side_effect=side_effect, autospec=True):
        ds.append(structures, buffer_size=2)

        # 5 items, buffer 2 => 2 chunks of 2 + 1 chunk of 1 => 3 writes
        assert append_handle.writelines.call_count == 3
