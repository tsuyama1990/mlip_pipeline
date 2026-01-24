from unittest.mock import MagicMock, patch

from ase import Atoms

from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.data_models.dft_models import DFTResult
from mlip_autopipec.dft.parsers import QEOutputParser
from mlip_autopipec.dft.recovery import DFTErrorType
from mlip_autopipec.dft.runner import QERunner


# Scenario 03-01: Generate Valid QE Input
def test_uat_03_01_generate_input(tmp_path):
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    (pseudo_dir / "Al.UPF").touch()

    config = DFTConfig(
        pseudopotential_dir=pseudo_dir,
        ecutwfc=30.0,
        kspacing=0.04,
        command="pw.x"
    )

    runner = QERunner(config)
    atoms = Atoms("Al", positions=[[0,0,0]], cell=[4,4,4], pbc=True)

    # We can inspect the private method or run the public one and mock execution
    with patch("mlip_autopipec.dft.runner.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        with patch("mlip_autopipec.dft.runner.QERunner._parse_output") as mock_parse:
             mock_parse.return_value = DFTResult(uid="1", energy=0, forces=[[0,0,0]], stress=[[0,0,0],[0,0,0],[0,0,0]], succeeded=True, wall_time=1, parameters={})
             with patch("shutil.which", return_value="/bin/pw.x"):
                 runner.run(atoms)

                 # Check if file was written
                 # The runner uses a temp dir, so we can't easily check the file unless we mock open or check args passed to subprocess

                 # Verify command args
                 args, kwargs = mock_run.call_args
                 cmd = args[0]
                 assert "pw.x" in cmd
                 assert "-in" in cmd
                 assert "pw.in" in cmd

# Scenario 03-02: Parse Successful Output
def test_uat_03_02_parse_output(tmp_path):
    # Create dummy file
    out_file = tmp_path / "pw.out"
    out_file.write_text("JOB DONE")

    # Mock ase read
    with patch("mlip_autopipec.dft.parsers.ase_read") as mock_read:
        atoms = Atoms("Al", positions=[[0,0,0]])
        atoms.calc = MagicMock()
        atoms.get_potential_energy = MagicMock(return_value=-10.0)
        atoms.get_forces = MagicMock(return_value=[[0.0, 0.0, 0.0]])
        atoms.get_stress = MagicMock(return_value=[[0,0,0],[0,0,0],[0,0,0]])
        mock_read.return_value = atoms

        parser = QEOutputParser(reader=mock_read)
        result = parser.parse(out_file, "uid", 1.0, {})

        assert result.succeeded
        assert result.energy == -10.0

# Scenario 03-03: Auto-Recovery
def test_uat_03_03_auto_recovery(tmp_path):
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    (pseudo_dir / "Al.UPF").touch()

    config = DFTConfig(
        pseudopotential_dir=pseudo_dir,
        ecutwfc=30.0,
        kspacing=0.04,
        command="pw.x",
        recoverable=True
    )

    runner = QERunner(config)
    atoms = Atoms("Al", positions=[[0,0,0]], cell=[4,4,4], pbc=True)

    with patch("shutil.which", return_value="/bin/pw.x"):
        with patch("mlip_autopipec.dft.runner.subprocess.run") as mock_run:
            # 1. Fail with convergence
            fail = MagicMock()
            fail.returncode = 1
            fail.stderr = "convergence NOT achieved"

            # 2. Success
            success = MagicMock()
            success.returncode = 0
            success.stderr = ""

            mock_run.side_effect = [fail, success]

            # Patch RecoveryHandler to simulate detection of convergence error
            with patch("mlip_autopipec.dft.runner.RecoveryHandler.analyze") as mock_analyze:
                mock_analyze.side_effect = [DFTErrorType.CONVERGENCE_FAIL, DFTErrorType.NONE]

                with patch("mlip_autopipec.dft.runner.QEOutputParser") as mock_parser:
                    mock_parser.return_value.parse.return_value = DFTResult(
                        uid="1", energy=0, forces=[[0,0,0]], stress=[[0,0,0],[0,0,0],[0,0,0]], succeeded=True, wall_time=1, parameters={}
                    )

                    # Inject the mock parser class into the runner instance
                    runner.parser_class = mock_parser

                    result = runner.run(atoms)

                    assert result.succeeded
                    assert mock_run.call_count == 2
