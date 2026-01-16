"""
This module contains handlers for resilience engineering, such as retry mechanisms.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class QERetryHandler:
    """Handles convergence errors in Quantum Espresso calculations."""

    def handle_convergence_error(
        self,
        log_content: str,
        current_params: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Diagnoses a convergence error and suggests modified parameters."""
        new_params = current_params.copy()
        if "convergence NOT achieved" in log_content:
            current_beta = new_params.get("mixing_beta", 0.7)
            new_beta = round(current_beta * 0.5, 2)
            if new_beta > 0.01:
                new_params["mixing_beta"] = new_beta
                logger.info(f"Convergence failed. Reducing mixing_beta to {new_beta}")
                return new_params

        if "Cholesky" in log_content:
            if new_params.get("diagonalization") != "cg":
                new_params["diagonalization"] = "cg"
                logger.info("Cholesky error detected. Switching to 'cg' diagonalization.")
                return new_params

        return None
