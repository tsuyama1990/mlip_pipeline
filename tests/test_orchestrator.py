import json
from pathlib import Path

import pytest

from mlip_autopipec.core.dataset import Dataset
from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models import GlobalConfig, Structure


def test_orchestrator_mock(tmp_path: Path) -> None:
    config = GlobalConfig(
        workdir=tmp_path / "test_run",
        max_cycles=2,
        generator={"type": "mock", "count": 2},
        oracle={"type": "mock"},
        trainer={"type": "mock"},
        dynamics={"type": "mock"}
    )

    orch = Orchestrator(config)
    orch.run()

    assert (tmp_path / "test_run" / "potential_cycle_0.yace").exists()
    assert (tmp_path / "test_run" / "potential_cycle_1.yace").exists()
    # Check dataset file
    dataset_file = tmp_path / "test_run" / "dataset.jsonl"
    assert dataset_file.exists()

    ds = Dataset(dataset_file)
    assert ds.count() > 0

def test_orchestrator_no_initial_structures(tmp_path: Path) -> None:
    config = GlobalConfig(
        workdir=tmp_path / "test_run_empty",
        max_cycles=2,
        generator={"type": "mock", "count": 0},
        oracle={"type": "mock"},
        trainer={"type": "mock"},
        dynamics={"type": "mock"}
    )

    orch = Orchestrator(config)
    orch.run()

    # Dataset should be empty
    ds = Dataset(tmp_path / "test_run_empty" / "dataset.jsonl")
    assert ds.count() == 0
    # No potential should be created
    assert not (tmp_path / "test_run_empty" / "potential_cycle_0.yace").exists()

def test_structure_validation_on_append(tmp_path: Path) -> None:
    ds = Dataset(tmp_path / "test.jsonl")
    s = Structure(
        atomic_numbers=[1],
        positions=[[0,0,0]],
        cell=[[1,0,0],[0,1,0],[0,0,1]],
        pbc=[True, True, True]
        # Missing energy/forces
    )
    # validate_labeled is called inside append
    with pytest.raises(ValueError, match="Structure must have energy"):
        ds.append([s])

def test_dataset_metadata(tmp_path: Path) -> None:
    ds = Dataset(tmp_path / "meta_test.jsonl")
    assert ds.count() == 0

    s = Structure(
        atomic_numbers=[1],
        positions=[[0,0,0]],
        cell=[[1,0,0],[0,1,0],[0,0,1]],
        pbc=[True, True, True],
        energy=-1.0,
        forces=[[0.0, 0.0, 0.0]]
    )

    ds.append([s])
    assert ds.count() == 1

    # Verify metadata file exists
    meta_file = tmp_path / "meta_test.meta.json"
    assert meta_file.exists()
    with meta_file.open() as f:
        meta = json.load(f)
        assert meta["count"] == 1

def test_dataset_scalability_mock(tmp_path: Path) -> None:
    """Verify that dataset handles append and count efficiently."""
    ds = Dataset(tmp_path / "scale_test.jsonl")

    # Simulate adding 100 structures
    structures = [
        Structure(
            atomic_numbers=[1],
            positions=[[0,0,0]],
            cell=[[1,0,0],[0,1,0],[0,0,1]],
            pbc=[True, True, True],
            energy=-1.0,
            forces=[[0.0, 0.0, 0.0]]
        ) for _ in range(100)
    ]

    ds.append(structures)
    assert ds.count() == 100

    # Iteration should work
    count = 0
    for _ in ds:
        count += 1
    assert count == 100
