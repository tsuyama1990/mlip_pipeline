import contextlib
import re
from pathlib import Path
from typing import Any, ClassVar

from mlip_autopipec.domain_models.dft_models import DFTErrorType


class DFTRetriableError(Exception):
    """Raised when an error is considered recoverable."""

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

    def __init__(self, config: Any):
        self.config = config

    def analyze_error(self, output_file: Path, stderr: str = "") -> DFTErrorType:
        """
        Analyzes the output file and stderr to determine the error type.
        """
        stdout = ""
        if output_file.exists():
            with contextlib.suppress(Exception):
                stdout = output_file.read_text(errors="replace")

        return self.analyze(stdout, stderr)

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

        # If no strategy found or fatal error, raise non-retriable exception
        # Actually QERunner catches DFTRetriableError to retry?
        # No, QERunner catches DFTRetriableError to CONTINUE (retry).
        # If we raise normal Exception, it breaks.
        msg = f"No recovery strategy available for {error_type} with params {current_params}"
        raise RuntimeError(msg)
