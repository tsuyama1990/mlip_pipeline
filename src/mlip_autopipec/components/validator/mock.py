import logging
import secrets

from mlip_autopipec.components.validator.base import BaseValidator
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.results import ValidationMetrics

logger = logging.getLogger(__name__)


class MockValidator(BaseValidator):
    def validate(self, potential: Potential) -> ValidationMetrics:
        logger.info("Validating potential")
        # Simulate validation metrics
        return ValidationMetrics(
            energy_rmse=0.01,
            force_rmse=0.05,
            stress_rmse=0.001,
            passed=True,
            details={
                "phonon_stability": True,
                "elastic_constants": {
                    "c11": 100 + secrets.randbelow(20),
                    "c12": 50 + secrets.randbelow(10),
                },
            },
        )
