from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.data_models.dft_models import DFTResult
from mlip_autopipec.dft.runner import DFTFatalError, QERunner


# Scenario 4.1: Successful Static Calculation
def test_uat_4_1_successful_calculation(tmp_path):
    (tmp_path / "pseudos").mkdir()
    config = DFTConfig(
        pseudopotential_dir=tmp_path / "pseudos",
        ecutwfc=30.0,
        kspacing=0.05,
        command="pw.x"
    )

    runner = QERunner(config)
    atoms = Atoms('Si2', positions=[[0,0,0], [1.5,0,0]], cell=[5,5,5], pbc=True)

    with patch("mlip_autopipec.dft.runner.subprocess.run") as mock_run, \
         patch("mlip_autopipec.dft.runner.shutil.which") as mock_which, \
         patch("mlip_autopipec.dft.runner.QEOutputParser") as mock_parser_cls:

        mock_which.return_value = "/bin/pw.x"

        # Simulate successful run writing JOB DONE
        def side_effect_success(*args, **kwargs):
            if kwargs.get('stdout'):
                kwargs['stdout'].write("... JOB DONE\n")
            return MagicMock(returncode=0, stderr="")

        mock_run.side_effect = side_effect_success

        # Parser returns success
        mock_parser = mock_parser_cls.return_value
        mock_parser.parse.return_value = DFTResult(
            uid="uat_4_1", energy=-100.0, forces=[[0,0,0], [0,0,0]],
            stress=[[0,0,0], [0,0,0], [0,0,0]], succeeded=True, wall_time=10.0,
            parameters={}, final_mixing_beta=0.7
        )

        result = runner.run(atoms, uid="uat_4_1")

        assert result.succeeded
        assert result.energy == -100.0


# Scenario 4.2: Convergence Failure Recovery
def test_uat_4_2_convergence_recovery(tmp_path):
    (tmp_path / "pseudos").mkdir()
    config = DFTConfig(
        pseudopotential_dir=tmp_path / "pseudos",
        ecutwfc=30.0,
        kspacing=0.05,
        command="pw.x"
    )

    runner = QERunner(config)
    atoms = Atoms('Fe', cell=[3,3,3], pbc=True)

    with patch("mlip_autopipec.dft.runner.subprocess.run") as mock_run, \
         patch("mlip_autopipec.dft.runner.shutil.which") as mock_which, \
         patch("mlip_autopipec.dft.runner.QEOutputParser") as mock_parser_cls, \
         patch("mlip_autopipec.dft.runner.RecoveryHandler") as mock_recovery:

        mock_which.return_value = "/bin/pw.x"

        call_count = 0
        def side_effect_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                if kwargs.get('stdout'):
                    kwargs['stdout'].write("convergence NOT achieved\n")
                    kwargs['stdout'].flush()
                return MagicMock(returncode=0, stderr="")
            if kwargs.get('stdout'):
                kwargs['stdout'].write("JOB DONE\n")
                kwargs['stdout'].flush()
            return MagicMock(returncode=0, stderr="")

        mock_run.side_effect = side_effect_run

        # Parser fails twice, then succeeds
        mock_parser = mock_parser_cls.return_value
        mock_parser.parse.side_effect = [
            Exception("Parse fail"),
            Exception("Parse fail"),
            DFTResult(
                uid="uat_4_2", energy=-50.0, forces=[[0,0,0]],
                stress=[[0,0,0], [0,0,0], [0,0,0]], succeeded=True, wall_time=20.0,
                parameters={"diagonalization": "cg"}, final_mixing_beta=0.3
            )
        ]

        def side_effect_analyze(stdout, stderr):
            if "convergence NOT achieved" in stdout:
                from mlip_autopipec.data_models.dft_models import DFTErrorType
                return DFTErrorType.CONVERGENCE_FAIL
            from mlip_autopipec.data_models.dft_models import DFTErrorType
            return DFTErrorType.NONE

        mock_recovery.analyze.side_effect = side_effect_analyze

        # Mock strategy
        def side_effect_strategy(error, params):
            new_p = params.copy()
            if params.get("mixing_beta", 0.7) > 0.35:
                new_p["mixing_beta"] = 0.3
            else:
                new_p["diagonalization"] = "cg"
            return new_p

        mock_recovery.get_strategy.side_effect = side_effect_strategy

        result = runner.run(atoms, uid="uat_4_2")

        assert result.succeeded
        assert mock_run.call_count == 3
        assert result.parameters["diagonalization"] == "cg"

# New Test: Recovery Strategy Failure (Max Retries)
def test_uat_4_2b_recovery_failure(tmp_path):
    (tmp_path / "pseudos").mkdir()
    config = DFTConfig(
        pseudopotential_dir=tmp_path / "pseudos",
        ecutwfc=30.0,
        kspacing=0.05,
        command="pw.x",
        max_retries=2 # Limit retries
    )

    runner = QERunner(config)
    atoms = Atoms('Fe', cell=[3,3,3], pbc=True)

    with patch("mlip_autopipec.dft.runner.subprocess.run") as mock_run, \
         patch("mlip_autopipec.dft.runner.shutil.which") as mock_which, \
         patch("mlip_autopipec.dft.runner.QEOutputParser") as mock_parser_cls, \
         patch("mlip_autopipec.dft.runner.RecoveryHandler") as mock_recovery:

        mock_which.return_value = "/bin/pw.x"

        # Always fail with convergence error
        def side_effect_run(*args, **kwargs):
            if kwargs.get('stdout'):
                kwargs['stdout'].write("convergence NOT achieved\n")
                kwargs['stdout'].flush()
            return MagicMock(returncode=0, stderr="")

        mock_run.side_effect = side_effect_run

        mock_parser = mock_parser_cls.return_value
        mock_parser.parse.side_effect = Exception("Parse fail")

        from mlip_autopipec.data_models.dft_models import DFTErrorType
        mock_recovery.analyze.return_value = DFTErrorType.CONVERGENCE_FAIL
        mock_recovery.get_strategy.return_value = {"mixing_beta": 0.1} # Dummy new param

        with pytest.raises(DFTFatalError, match="failed after 3 attempts"):
            # 1 initial + 2 retries = 3 total attempts
            runner.run(atoms, uid="uat_4_2b")


# Scenario 4.3: Fatal Error Handling
def test_uat_4_3_fatal_error(tmp_path):
    (tmp_path / "pseudos").mkdir()
    config = DFTConfig(
        pseudopotential_dir=tmp_path / "pseudos",
        ecutwfc=30.0,
        kspacing=0.05,
        command="pw.x"
    )

    runner = QERunner(config)
    atoms = Atoms('H', cell=[5,5,5], pbc=True)

    with patch("mlip_autopipec.dft.runner.subprocess.run") as mock_run, \
         patch("mlip_autopipec.dft.runner.shutil.which") as mock_which, \
         patch("mlip_autopipec.dft.runner.QEOutputParser") as mock_parser_cls, \
         patch("mlip_autopipec.dft.runner.RecoveryHandler") as mock_recovery:

        mock_which.return_value = "/bin/pw.x"

        def segfault(*args, **kwargs):
             return MagicMock(returncode=139, stderr="Segmentation fault")

        mock_run.side_effect = segfault
        mock_parser = mock_parser_cls.return_value
        mock_parser.parse.side_effect = Exception("Crash")

        from mlip_autopipec.data_models.dft_models import DFTErrorType
        mock_recovery.analyze.return_value = DFTErrorType.NONE

        with pytest.raises(DFTFatalError, match="exited with 139"):
            runner.run(atoms, uid="uat_4_3")
