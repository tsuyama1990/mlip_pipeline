import pytest

from mlip_autopipec.data_models.dft_models import DFTErrorType
from mlip_autopipec.dft.recovery import RecoveryHandler


def test_analyze_convergence_error():
    stdout = "some output\nconvergence NOT achieved after 100 iterations"
    stderr = ""
    error = RecoveryHandler.analyze(stdout, stderr)
    assert error == DFTErrorType.CONVERGENCE_FAIL

def test_analyze_diagonalization_error():
    stdout = "c_bands:  1 eigenvalues not converged"
    stderr = "error in diagonalization"
    error = RecoveryHandler.analyze(stdout, stderr)
    assert error == DFTErrorType.DIAGONALIZATION_ERROR

def test_analyze_oom_error():
    stdout = ""
    stderr = "slurm step: error: ... oom-kill event"
    error = RecoveryHandler.analyze(stdout, stderr)
    assert error == DFTErrorType.OOM_KILL

def test_analyze_timeout_error():
    stdout = "Maximum CPU time exceeded"
    stderr = ""
    error = RecoveryHandler.analyze(stdout, stderr)
    assert error == DFTErrorType.MAX_CPU_TIME

def test_analyze_no_error():
    stdout = "JOB DONE"
    stderr = ""
    error = RecoveryHandler.analyze(stdout, stderr)
    assert error == DFTErrorType.NONE

def test_recovery_strategy_ladder():
    # Initial state
    params = {"mixing_beta": 0.7, "diagonalization": "david", "degauss": 0.02}

    # Step 1: Convergence Fail -> Reduce Beta
    new_params = RecoveryHandler.get_strategy(DFTErrorType.CONVERGENCE_FAIL, params)
    assert new_params["mixing_beta"] == 0.3
    assert new_params["diagonalization"] == "david" # Unchanged

    # Step 2: Still Failing (beta=0.3) -> Robust Solver
    params_level_1 = {"mixing_beta": 0.3, "diagonalization": "david", "degauss": 0.02}
    new_params_2 = RecoveryHandler.get_strategy(DFTErrorType.CONVERGENCE_FAIL, params_level_1)
    assert new_params_2["diagonalization"] == "cg"
    assert new_params_2["mixing_beta"] == 0.3

    # Step 3: Still Failing (beta=0.3, diag=cg) -> High Temp
    params_level_2 = {"mixing_beta": 0.3, "diagonalization": "cg", "degauss": 0.02}
    new_params_3 = RecoveryHandler.get_strategy(DFTErrorType.CONVERGENCE_FAIL, params_level_2)
    assert new_params_3["degauss"] == 0.03

def test_recovery_diagonalization_error():
    # Immediate switch to CG
    params = {"mixing_beta": 0.7, "diagonalization": "david"}
    new_params = RecoveryHandler.get_strategy(DFTErrorType.DIAGONALIZATION_ERROR, params)
    assert new_params["diagonalization"] == "cg"

def test_recovery_unknown_error():
    params = {}
    with pytest.raises(Exception, match="No recovery strategy"):
        RecoveryHandler.get_strategy(DFTErrorType.OOM_KILL, params)
