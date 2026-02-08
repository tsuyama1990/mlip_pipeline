from pathlib import Path

import numpy as np
import pytest

from mlip_autopipec.core.dataset import Dataset
from mlip_autopipec.domain_models import Structure


def create_dummy_structure(idx: int) -> Structure:
    return Structure(
        positions=np.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=np.array([1]),
        cell=np.eye(3),
        energy=-1.0,
        forces=np.zeros((1, 3)),
        stress=np.zeros((3, 3)),
        properties={"idx": idx},
    )


def test_dataset_init(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    dataset = Dataset(path)
    assert len(dataset) == 0
    assert dataset.path == path.resolve()
    assert dataset.meta_path.exists()


def test_dataset_append_streaming(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    dataset = Dataset(path)

    structures = [create_dummy_structure(i) for i in range(10)]
    dataset.append(structures, batch_size=2)

    assert len(dataset) == 10

    # Verify content
    loaded = list(dataset)
    assert len(loaded) == 10
    assert loaded[0].properties is not None
    assert loaded[0].properties["idx"] == 0
    assert loaded[9].properties is not None
    assert loaded[9].properties["idx"] == 9


def test_dataset_security_check(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()

    # Path inside root
    safe_path = root / "data.jsonl"
    Dataset(safe_path, root_dir=root)

    # Path outside root
    unsafe_path = tmp_path / "outside.jsonl"
    with pytest.raises(ValueError, match="outside root directory"):
        Dataset(unsafe_path, root_dir=root)


def test_dataset_corrupt_line(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "corrupt.jsonl"
    path.write_text('{"valid": "json"}\nINVALID_JSON\n{"valid": "json"}')

    # Just write dummy meta so init doesn't fail
    dataset = Dataset(path)
    # Mock meta count or just rely on iter not using count
    # Dataset iter reads file.

    # We need to construct valid structures for the valid lines for it to yield
    # Actually, {"valid": "json"} will fail validation if passed to Structure constructor?
    # Dataset iter: yields Structure(**data).
    # So valid json but invalid structure raises ValidationError.
    # But iter only catches JSONDecodeError.
    # Let's write valid structure JSON

    s = create_dummy_structure(0)
    valid_json = s.model_dump_json()

    path.write_text(f"{valid_json}\nBROKEN_JSON\n{valid_json}")

    items = list(dataset)
    assert len(items) == 2
    assert "Corrupt dataset line" in caplog.text
