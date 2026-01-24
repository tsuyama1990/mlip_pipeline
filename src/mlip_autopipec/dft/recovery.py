import re
from typing import Any

from mlip_autopipec.data_models.dft_models import DFTErrorType


class RecoveryHandler:
    """
    Analyzes DFT errors and suggests recovery strategies.
    """

    PATTERNS = {
        DFTErrorType.DIAGONALIZATION_ERROR: [
            re.compile(r"error in diagonalization", re.IGNORECASE),
            re.compile(r"cholesky decomposition failed", re.IGNORECASE),
            re.compile(r"c_bands:.*eigenvalues not converged", re.IGNORECASE),
        ],
        DFTErrorType.CONVERGENCE_FAIL: [
            re.compile(r"convergence NOT achieved", re.IGNORECASE),
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
        # Deep copy to avoid mutating original if needed
        new_params = current_params.copy()

        # Ensure defaults are populated for logic check
        beta = new_params.get("mixing_beta", 0.7)
        diag = new_params.get("diagonalization", "david")
        degauss = new_params.get("degauss", 0.02)

        if error_type == DFTErrorType.CONVERGENCE_FAIL:
            # Level 1: Reduce Mixing
            if beta > 0.35:
                new_params["mixing_beta"] = 0.3
                return new_params

            # Level 2: Robust Solver
            if diag == "david":
                new_params["diagonalization"] = "cg"
                return new_params

            # Level 3: High Temperature
            # We increase degauss in small steps
            new_params["degauss"] = round(degauss + 0.01, 4)
            return new_params

        if error_type == DFTErrorType.DIAGONALIZATION_ERROR:
            # Immediate switch to CG
            new_params["diagonalization"] = "cg"
            return new_params

        # If no strategy found or fatal error
        raise Exception(
            f"No recovery strategy available for {error_type} with params {current_params}"
        )
