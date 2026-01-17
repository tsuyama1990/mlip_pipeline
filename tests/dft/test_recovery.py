import pytest

from mlip_autopipec.data_models.dft_models import DFTErrorType
from mlip_autopipec.dft.recovery import RecoveryHandler


def test_recovery_analysis_convergence():
    stdout = """
    some log lines
    convergence NOT achieved
    some other lines
    """
    error_type = RecoveryHandler.analyze(stdout, "")
    assert error_type == DFTErrorType.CONVERGENCE_FAIL


def test_recovery_analysis_diagonalization():
    stdout = """
    some log lines
    error in diagonalization
    """
    error_type = RecoveryHandler.analyze(stdout, "")
    assert error_type == DFTErrorType.DIAGONALIZATION_ERROR


def test_recovery_strategy_mixing():
    # Initial state
    current_params = {"mixing_beta": 0.7, "electron_maxstep": 100}

    new_params = RecoveryHandler.get_strategy(DFTErrorType.CONVERGENCE_FAIL, current_params)

    assert new_params["mixing_beta"] == 0.3
    # Check that other params are preserved or updated
    assert new_params["electron_maxstep"] == 100  # unless changed


def test_recovery_strategy_fatal():
    # If we run out of strategies or hit a fatal error
    current_params = {}
    with pytest.raises(Exception):  # Or return None/Failure
        RecoveryHandler.get_strategy(DFTErrorType.OOM_KILL, current_params)
