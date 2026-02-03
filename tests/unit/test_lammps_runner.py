from pathlib import Path
from unittest.mock import MagicMock, patch

from ase import Atoms

from mlip_autopipec.config.config_model import LammpsConfig
from mlip_autopipec.domain_models.dynamics import MDResult, MDStatus
from mlip_autopipec.physics.dynamics.lammps_runner import LammpsRunner


@patch("mlip_autopipec.physics.dynamics.lammps_runner.subprocess.run")
def test_runner_execution(mock_run: MagicMock, tmp_path: Path) -> None:
    # Setup
    config = LammpsConfig(command="lmp_serial")
    runner = LammpsRunner(config)
    atoms = Atoms("H")
    potential_path = Path("test.yace")
    work_dir = tmp_path / "test_work_dir"

    # Mock subprocess
    mock_run.return_value.returncode = 0

    # Act
    with patch("mlip_autopipec.physics.dynamics.lammps_runner.LammpsInputGenerator") as mock_gen, \
         patch("mlip_autopipec.physics.dynamics.lammps_runner.LogParser") as mock_parser, \
         patch("mlip_autopipec.physics.dynamics.lammps_runner.write") as mock_write:

         mock_gen.return_value.generate_input.return_value = "input data"

         mock_result = MDResult(status=MDStatus.COMPLETED)
         mock_parser.return_value.parse.return_value = mock_result

         result = runner.run(atoms, potential_path, work_dir, parameters={})

         assert result.status == MDStatus.COMPLETED
         mock_run.assert_called()
         mock_gen.assert_called()
         mock_parser.assert_called()
         mock_write.assert_called() # Checking that atoms are written to data file
