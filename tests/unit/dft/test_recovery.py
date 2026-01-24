from mlip_autopipec.data_models.dft_models import DFTErrorType
from mlip_autopipec.dft.recovery import RecoveryHandler


def test_analyze_convergence_fail():
    stdout = "some output"
    stderr = "convergence NOT achieved after 100 iterations"
    assert RecoveryHandler.analyze(stdout, stderr) == DFTErrorType.CONVERGENCE_FAIL

def test_analyze_diagonalization_error():
    stdout = "error in diagonalization"
    stderr = ""
    assert RecoveryHandler.analyze(stdout, stderr) == DFTErrorType.DIAGONALIZATION_ERROR

def test_analyze_unknown():
    stdout = "Everything is fine"
    stderr = "Just a warning"
    assert RecoveryHandler.analyze(stdout, stderr) == DFTErrorType.NONE

def test_strategy_convergence():
    params = {"mixing_beta": 0.7, "diagonalization": "david", "degauss": 0.02}

    # Step 1: Reduce beta
    new_params = RecoveryHandler.get_strategy(DFTErrorType.CONVERGENCE_FAIL, params)
    assert new_params["mixing_beta"] == 0.3

    # Step 2: Change diag (if beta already low)
    params["mixing_beta"] = 0.3
    new_params = RecoveryHandler.get_strategy(DFTErrorType.CONVERGENCE_FAIL, params)
    assert new_params["diagonalization"] == "cg"

    # Step 3: Increase temp (if beta low and diag cg)
    params["diagonalization"] = "cg"
    new_params = RecoveryHandler.get_strategy(DFTErrorType.CONVERGENCE_FAIL, params)
    assert new_params["degauss"] == 0.03
