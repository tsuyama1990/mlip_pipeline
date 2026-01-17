# Cycle 02 UAT
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from ase import Atoms

from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.data_models.dft_models import DFTErrorType, DFTResult
from mlip_autopipec.dft.runner import QERunner


# Helpers for UAT (can be shared or inline)
def create_mock_atoms(symbol="Al", positions=None, cell=None):
    if positions is None:
        positions = [[0, 0, 0]]
    if cell is None:
        cell = [4, 4, 4]
    return Atoms(symbol, positions=positions, cell=cell, pbc=True)


# UAT-02-01: Standard SCF Calculation
def test_uat_02_01_standard_scf():
    print("\n--- UAT-02-01: Standard SCF Calculation ---")
    config = DFTConfig(command="pw.x", pseudo_dir=Path("/tmp"), timeout=100)
    runner = QERunner(config)
    atoms = create_mock_atoms()

    # Mock everything
    with (
        patch("subprocess.run") as mock_run,
        patch("mlip_autopipec.dft.runner.InputGenerator.create_input_string") as mock_input,
        patch("mlip_autopipec.dft.runner.QERunner._parse_output") as mock_parse,
    ):
        mock_input.return_value = "pw.in content"

        proc = MagicMock()
        proc.returncode = 0
        mock_run.return_value = proc

        expected_result = DFTResult(
            uid="uat-01",
            energy=-100.0,
            forces=[[0.0, 0.0, 0.0]],
            stress=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            succeeded=True,
            wall_time=10.0,
            parameters={},
            final_mixing_beta=0.7,
        )
        mock_parse.return_value = expected_result

        result = runner.run(atoms, uid="uat-01")

        assert result.succeeded
        assert result.energy == -100.0
        print("✅ UAT-02-01 Passed")


# UAT-02-02: Convergence Auto-Recovery
def test_uat_02_02_auto_recovery():
    print("\n--- UAT-02-02: Convergence Auto-Recovery ---")
    config = DFTConfig(command="pw.x", pseudo_dir=Path("/tmp"), recoverable=True)
    runner = QERunner(config)
    atoms = create_mock_atoms()

    with (
        patch("subprocess.run") as mock_run,
        patch("mlip_autopipec.dft.runner.InputGenerator.create_input_string") as mock_input,
        patch("mlip_autopipec.dft.recovery.RecoveryHandler.analyze") as mock_analyze,
        patch("mlip_autopipec.dft.recovery.RecoveryHandler.get_strategy") as mock_strategy,
        patch("mlip_autopipec.dft.runner.QERunner._parse_output") as mock_parse,
    ):
        # 1st run fails
        proc_fail = MagicMock()
        proc_fail.returncode = 1
        proc_fail.stdout = "Error"

        # 2nd run succeeds
        proc_success = MagicMock()
        proc_success.returncode = 0
        proc_success.stdout = "Done"

        mock_run.side_effect = [proc_fail, proc_success]

        mock_analyze.return_value = DFTErrorType.CONVERGENCE_FAIL
        mock_strategy.return_value = {"mixing_beta": 0.3}

        # Capture the input params passed to generator
        mock_input.side_effect = lambda atoms, params: f"input with {params}"

        # Parse output mock: 1st time fails (raises), 2nd time succeeds
        expected_result = DFTResult(
            uid="uat-02",
            energy=-100.0,
            forces=[[0, 0, 0]],
            stress=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            succeeded=True,
            wall_time=10,
            parameters={"mixing_beta": 0.3},
            final_mixing_beta=0.3,
        )
        mock_parse.side_effect = [Exception("Fail"), expected_result]

        result = runner.run(atoms, uid="uat-02")

        assert result.succeeded
        assert result.final_mixing_beta == 0.3
        assert mock_run.call_count == 2

        # IMPORTANT: Verify params actually changed
        # mock_input call args: (atoms, params)
        # Call 1: params={} (default)
        # Call 2: params={'mixing_beta': 0.3}
        args1 = mock_input.call_args_list[0]
        args2 = mock_input.call_args_list[1]

        assert args1[0][1] == {}
        assert args2[0][1] == {"mixing_beta": 0.3}

        print("✅ UAT-02-02 Passed")


# UAT-02-03: Magnetic System Handling
def test_uat_02_03_magnetic_handling():
    print("\n--- UAT-02-03: Magnetic System Handling ---")
    # Using real InputGenerator logic here, no mock for create_input_string
    from mlip_autopipec.dft.inputs import InputGenerator

    atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=[2.5, 2.5, 2.5], pbc=True)
    input_str = InputGenerator.create_input_string(atoms)

    # Check for magnetic flags
    assert "nspin" in input_str
    # Note: Depending on ASE version/implementation, exact string might vary, but key words must exist
    assert "starting_magnetization" in input_str
    print("✅ UAT-02-03 Passed")


if __name__ == "__main__":
    # Manually run if executed as script
    try:
        test_uat_02_01_standard_scf()
        test_uat_02_02_auto_recovery()
        test_uat_02_03_magnetic_handling()
        print("\n🎉 ALL UAT Passed!")
    except Exception as e:
        print(f"\n❌ UAT Failed: {e}")
        sys.exit(1)
