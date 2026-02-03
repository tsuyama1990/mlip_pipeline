from pathlib import Path
from unittest.mock import MagicMock

from ase import Atoms
from ase.io import write

from mlip_autopipec.domain_models.dynamics import MDResult, MDStatus
from mlip_autopipec.domain_models.exploration import ExplorationMethod, ExplorationTask
from mlip_autopipec.orchestration.otf_loop import OTFLoop


def test_otf_loop_halted(temp_dir: Path) -> None:
    # Setup
    runner_mock = MagicMock()
    loop = OTFLoop(runner_mock)

    seed = Atoms("Cu", positions=[[0, 0, 0]], cell=[3, 3, 3])

    # Create a fake trajectory file
    traj_path = temp_dir / "traj.xyz"
    write(traj_path, [seed, seed])  # 2 frames

    # Mock result
    result = MDResult(
        status=MDStatus.HALTED, trajectory_path=traj_path, halt_step=100
    )
    runner_mock.run.return_value = result

    task = ExplorationTask(method=ExplorationMethod.MD)

    # Execute
    candidates = loop.execute_task(task, seed, None, temp_dir)

    # Verify
    # 1 original halted + 20 local = 21
    assert len(candidates) == 21

    # Check metadata
    first = candidates[0]
    assert first.metadata.generation_method == "md_halted"
    assert first.metadata.source == "otf_loop"

    second = candidates[1]
    assert second.metadata.generation_method == "random_displacement_halt"
    assert second.metadata.parent_structure_id == "halted_step_100"
    assert second.metadata.source == "otf_loop"


def test_otf_loop_completed(temp_dir: Path) -> None:
    # Setup
    runner_mock = MagicMock()
    loop = OTFLoop(runner_mock)

    seed = Atoms("Cu", positions=[[0, 0, 0]], cell=[3, 3, 3])

    # Mock result
    result = MDResult(status=MDStatus.COMPLETED)
    runner_mock.run.return_value = result

    task = ExplorationTask(method=ExplorationMethod.MD)

    # Execute
    candidates = loop.execute_task(task, seed, None, temp_dir)

    # Verify
    assert len(candidates) == 0
