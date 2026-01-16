import logging
from typing import Any

from mlip_autopipec.config.models import DFTJob

logger = logging.getLogger(__name__)

def dft_retry_handler(exception: Exception, kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """
    Handles specific DFT convergence errors by suggesting modified parameters.
    """
    log_content = getattr(exception, "stdout", "") + getattr(exception, "stderr", "")
    job = kwargs.get("job")
    if not isinstance(job, DFTJob):
        return None

    current_params = job.params.model_copy()
    updated_fields = {}

    if "convergence NOT achieved" in log_content and current_params.mixing_beta > 0.1:
        updated_fields["mixing_beta"] = current_params.mixing_beta / 2
        logger.info(f"Convergence failed. Halving mixing_beta to {updated_fields['mixing_beta']}.")

    if "Cholesky" in log_content and current_params.diagonalization == "david":
        updated_fields["diagonalization"] = "cg"
        logger.info("Cholesky issue detected. Switching diagonalization to 'cg'.")

    if updated_fields:
        # Create new params object with updated fields
        new_params = current_params.model_copy(update=updated_fields)
        job.params = new_params
        return {"job": job}

    return None
