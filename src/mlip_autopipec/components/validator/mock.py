import logging
import secrets
from typing import Any

import numpy as np

from mlip_autopipec.domain_models import Potential
from mlip_autopipec.interfaces import BaseValidator

logger = logging.getLogger(__name__)


class MockValidator(BaseValidator):
    """
    Mock implementation of a validator.
    Returns dummy validation metrics.
    """

    def __init__(self, fail_rate: float = 0.0, **kwargs: Any) -> None:
        """
        Args:
            fail_rate: Probability of failure during validation (0.0 to 1.0).
            **kwargs: Ignored extra arguments.
        """
        self.fail_rate = fail_rate
        self.rng = np.random.default_rng(secrets.randbits(128))
        if kwargs:
            logger.debug(f"MockValidator received extra args: {kwargs}")

    def validate(self, potential: Potential) -> dict[str, Any]:
        """
        Returns dummy validation metrics for the potential.
        """
        if self.rng.random() < self.fail_rate:
            msg = "MockValidator: Simulated failure during validation."
            raise RuntimeError(msg)

        logger.info(f"MockValidator: Validating potential {potential.version}...")

        return {
            "test_mae_e": 0.002,
            "test_mae_f": 0.02,
            "validation_passed": True,
        }
