import re
from enum import Enum
from typing import Any, ClassVar


class DFTErrorType(str, Enum):
    CONVERGENCE_FAIL = "CONVERGENCE_FAIL"
    DIAGONALIZATION_ERROR = "DIAGONALIZATION_ERROR"
    MAX_CPU_TIME = "MAX_CPU_TIME"
    OOM_KILL = "OOM_KILL"
    UNKNOWN = "UNKNOWN"
    NONE = "NONE"

class RecoveryHandler:
    """
    Analyzes DFT output/error logs and suggests recovery strategies.
    """

    PATTERNS: ClassVar[dict[DFTErrorType, list[re.Pattern[str]]]] = {
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
        Returns updated parameters based on the error type.
        """
        new_params = current_params.copy()

        if error_type == DFTErrorType.CONVERGENCE_FAIL:
            beta = new_params.get("mixing_beta", 0.7)
            diag = new_params.get("diagonalization", "david")

            # Strategy 1: Reduce Beta if high
            if beta > 0.3:
                new_params["mixing_beta"] = 0.3
                return new_params

            # Strategy 2: Switch diagonalization if beta is low but diag is david
            if diag == "david":
                new_params["diagonalization"] = "cg"
                return new_params

            # Strategy 3: Increase Temperature (Smearing)
            degauss = new_params.get("degauss", 0.02)
            new_params["degauss"] = degauss + 0.01
            return new_params

        if error_type == DFTErrorType.DIAGONALIZATION_ERROR:
            # Switch algo
            algo = new_params.get("diagonalization", "david")
            if algo == "david":
                new_params["diagonalization"] = "cg"
            return new_params

        # If no strategy found or fatal error
        msg = f"No recovery strategy available for {error_type} with params {current_params}"
        raise RuntimeError(msg)
