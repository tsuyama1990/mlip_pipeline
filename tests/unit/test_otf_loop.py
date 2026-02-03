from pathlib import Path
from unittest.mock import MagicMock

import pytest
from ase import Atoms
from ase.io import write

from mlip_autopipec.domain_models.dynamics import MDResult, MDStatus
from mlip_autopipec.domain_models.exploration import ExplorationMethod, ExplorationTask
from mlip_autopipec.orchestration.otf_loop import OTFLoop
from mlip_autopipec.physics.dynamics.lammps_runner import LammpsRunner


@pytest.fixture
def mock_runner() -> MagicMock:
    return MagicMock(spec=LammpsRunner)


def test_otf_loop_completed(mock_runner: MagicMock, temp_dir: Path) -> None:
    # Setup
    mock_runner.run.return_value = MDResult(status=MDStatus.COMPLETED)
    otf_loop = OTFLoop(mock_runner)

    seed = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]], cell=[10, 10, 10])
    task = ExplorationTask(method=ExplorationMethod.MD, parameters={})

    # Run
    candidates = otf_loop.execute_task(task, seed, None, temp_dir)

    # Assert
    assert len(candidates) == 0
    mock_runner.run.assert_called_once()


def test_otf_loop_halted(mock_runner: MagicMock, temp_dir: Path) -> None:
    # Setup
    traj_path = temp_dir / "traj.xyz"

    # Create a dummy trajectory file
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[5, 5, 5], pbc=True)
    write(traj_path, atoms)

    mock_runner.run.return_value = MDResult(
        status=MDStatus.HALTED, halt_step=100, trajectory_path=traj_path
    )

    otf_loop = OTFLoop(mock_runner)

    seed = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[5, 5, 5], pbc=True)
    task = ExplorationTask(method=ExplorationMethod.MD, parameters={"local_sampling_count": 3})

    # Run
    candidates = otf_loop.execute_task(task, seed, None, temp_dir)

    # Assert
    # Should have 1 (anchor) + 3 (local samples) = 4 candidates
    assert len(candidates) == 4

    # Check metadata
    anchor = candidates[0]
    assert anchor.metadata.source == "md_halt"
    assert anchor.metadata.generation_method == "embedding"

    sample = candidates[1]
    assert sample.metadata.source == "md_halt_local_search"
    assert sample.metadata.generation_method == "random_displacement"
    assert sample.metadata.parent_structure_id == "halted_anchor"
