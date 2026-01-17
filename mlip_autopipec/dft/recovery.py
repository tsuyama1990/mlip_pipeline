import re
from typing import Any

from mlip_autopipec.data_models.dft_models import DFTErrorType


class RecoveryHandler:
    """
    Analyzes DFT errors and suggests recovery strategies.
    """

    PATTERNS = {
        DFTErrorType.CONVERGENCE_FAIL: [
            re.compile(r"convergence NOT achieved", re.IGNORECASE),
            re.compile(r"c_bands:.*eigenvalues not converged", re.IGNORECASE),
        ],
        DFTErrorType.DIAGONALIZATION_ERROR: [
            re.compile(r"error in diagonalization", re.IGNORECASE),
            re.compile(r"cholesky decomposition failed", re.IGNORECASE),
        ],
        DFTErrorType.MAX_CPU_TIME: [re.compile(r"maximum CPU time exceeded", re.IGNORECASE)],
        DFTErrorType.OOM_KILL: [
            re.compile(r"oom-kill", re.IGNORECASE),
            re.compile(r"out of memory", re.IGNORECASE),
        ],
    }

    @staticmethod
    def analyze(stdout: str, stderr: str) -> DFTErrorType:
        """
        Scans stdout/stderr for known error patterns.
        """
        combined = stdout + "\n" + stderr

        for error_type, patterns in RecoveryHandler.PATTERNS.items():
            for pattern in patterns:
                if pattern.search(combined):
                    return error_type

        return DFTErrorType.NONE

    @staticmethod
    def get_strategy(error_type: DFTErrorType, current_params: dict[str, Any]) -> dict[str, Any]:
        """
        Returns updated parameters based on the error type and current state.
        This implements the 'Recovery Tree'.
        """
        # Deep copy to avoid mutating original if needed, but dict copy is enough for top level
        new_params = current_params.copy()

        if error_type == DFTErrorType.CONVERGENCE_FAIL:
            # Strategy: Reduce mixing beta -> Local-TF -> Increase Temp

            current_beta = new_params.get("mixing_beta", 0.7)
            current_mode = (
                new_params.get("input_data", {}).get("electrons", {}).get("mixing_mode", "plain")
            )
            current_degauss = new_params.get("degauss", 0.02)

            if current_beta > 0.35:
                # Step 1: Reduce beta
                new_params["mixing_beta"] = 0.3
                return new_params

            # If beta is already low, try changing mixing mode?
            # Or just check if we have tried other things.
            # A robust state machine might store 'recovery_stage' in params.

            # Simple stateless logic based on values:
            if current_beta <= 0.3:
                # Try increasing temperature
                new_params["degauss"] = 0.05
                # new_params["mixing_beta"] = 0.1 # even lower?
                return new_params

        elif error_type == DFTErrorType.DIAGONALIZATION_ERROR:
            new_params["diagonalization"] = "cg"
            return new_params

        # If no strategy found or fatal error
        raise Exception(
            f"No recovery strategy available for {error_type} with params {current_params}"
        )
