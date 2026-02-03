from pathlib import Path
from unittest.mock import MagicMock, patch

from ase import Atoms

from mlip_autopipec.domain_models.dynamics import MDResult, MDStatus
from mlip_autopipec.domain_models.exploration import ExplorationMethod, ExplorationTask
from mlip_autopipec.orchestration.otf_loop import OTFLoop


def test_otf_loop_halt_handling(tmp_path: Path) -> None:
    # Create dummy trajectory file
    traj_path = tmp_path / "traj.lammpstrj"
    traj_path.touch()

    # Mock Runner
    mock_runner = MagicMock()
    mock_runner.run.return_value = MDResult(
        status=MDStatus.HALTED,
        halt_step=50,
        trajectory_path=traj_path
    )

    loop = OTFLoop(runner=mock_runner)

    task = ExplorationTask(method=ExplorationMethod.MD)
    atoms = Atoms("H")
    potential_path = Path("pot.yace")
    work_dir = tmp_path

    # Mock ase.io.read to return a dummy structure for the halted frame
    dummy_halted_struct = Atoms("H", positions=[[1, 1, 1]])

    # We use patch on the module where read is imported.
    # Assuming otf_loop imports read from ase.io
    with patch("mlip_autopipec.orchestration.otf_loop.read") as mock_read, \
         patch("mlip_autopipec.orchestration.otf_loop.write") as mock_write:

        # read() can return list or Atoms.
        mock_read.return_value = dummy_halted_struct

        candidates = loop.execute_task(task, atoms, potential_path, work_dir)

        assert len(candidates) == 1
        assert candidates[0].metadata.generation_method == "md_halted"
        mock_read.assert_called()
        mock_write.assert_called()
