import logging
import secrets
from typing import Any

from mlip_autopipec.components.validator.base import BaseValidator
from mlip_autopipec.domain_models.potential import Potential

logger = logging.getLogger(__name__)


class MockValidator(BaseValidator):
    def validate(self, potential: Potential) -> dict[str, Any]:
        logger.info("Validating potential")
        # Simulate validation metrics
        return {
            "phonon_stability": True,
            "elastic_constants": {
                "c11": 100 + secrets.randbelow(20),
                "c12": 50 + secrets.randbelow(10),
            },
            "passed": True,
        }
