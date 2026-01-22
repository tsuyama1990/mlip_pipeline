import pytest
from mlip_autopipec.dft.recovery import RecoveryHandler
from mlip_autopipec.data_models.dft_models import DFTErrorType

def test_analyze_convergence_error():
    stdout = "some output"
    stderr = "Error: convergence NOT achieved after 100 iterations"
    error = RecoveryHandler.analyze(stdout, stderr)
    assert error == DFTErrorType.CONVERGENCE_FAIL

def test_analyze_diagonalization_error():
    stdout = "c_bands:  1 eigenvalues not converged"
    stderr = "error in diagonalization"
    error = RecoveryHandler.analyze(stdout, stderr)
    assert error == DFTErrorType.DIAGONALIZATION_ERROR

def test_recovery_strategy_ladder():
    # Initial state
    params = {"mixing_beta": 0.7, "diagonalization": "david", "degauss": 0.02}

    # Step 1: Convergence Fail -> Reduce Beta
    new_params = RecoveryHandler.get_strategy(DFTErrorType.CONVERGENCE_FAIL, params)
    assert new_params["mixing_beta"] == 0.3

    # Step 2: Still Convergence Fail -> Increase Temp (if beta is already low)
    # The existing logic I read was: if beta <= 0.3 -> increase temp.
    # Wait, SPEC says:
    # Level 1: Beta 0.3
    # Level 2: Robust Solver (CG) (for Convergence Fail? Or just general?)
    # SPEC says: "Level 2 (Still Failing): Robust Solver. Change solver to diagonalization='cg'"
    # So if beta is 0.3 and it fails again with Convergence Fail, we switch to CG.

    params_level_1 = {"mixing_beta": 0.3, "diagonalization": "david", "degauss": 0.02}
    new_params_2 = RecoveryHandler.get_strategy(DFTErrorType.CONVERGENCE_FAIL, params_level_1)
    # This implies the logic should check if beta is low, then try CG.
    assert new_params_2["diagonalization"] == "cg"

    # Step 3: Still Failing -> High Temp
    params_level_2 = {"mixing_beta": 0.3, "diagonalization": "cg", "degauss": 0.02}
    new_params_3 = RecoveryHandler.get_strategy(DFTErrorType.CONVERGENCE_FAIL, params_level_2)
    assert new_params_3["degauss"] == 0.03 # degauss += 0.01

def test_recovery_diagonalization_error():
    # If explicit diagonalization error, switch to CG immediately
    params = {"mixing_beta": 0.7, "diagonalization": "david"}
    new_params = RecoveryHandler.get_strategy(DFTErrorType.DIAGONALIZATION_ERROR, params)
    assert new_params["diagonalization"] == "cg"
