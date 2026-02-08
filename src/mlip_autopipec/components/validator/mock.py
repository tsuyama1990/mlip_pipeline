import logging
from typing import Any

from mlip_autopipec.domain_models import Potential
from mlip_autopipec.interfaces import BaseValidator

logger = logging.getLogger(__name__)


class MockValidator(BaseValidator):
    """
    Mock implementation of a validator.
    Returns dummy validation metrics.
    """

    def validate(self, potential: Potential) -> dict[str, Any]:
        """
        Returns dummy validation metrics for the potential.
        """
        logger.info(f"MockValidator: Validating potential {potential.version}...")

        return {
            "test_mae_e": 0.002,
            "test_mae_f": 0.02,
            "validation_passed": True,
        }
