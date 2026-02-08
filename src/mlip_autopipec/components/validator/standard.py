import logging

from mlip_autopipec.components.validator.base import BaseValidator
from mlip_autopipec.domain_models.config import StandardValidatorConfig
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.results import ValidationMetrics

logger = logging.getLogger(__name__)


class StandardValidator(BaseValidator):
    """
    Standard implementation of the Validator component.

    This component validates the quality of a potential by calculating various physical
    properties (Phonons, EOS, Elastic constants) and comparing them against reference data.
    """

    def __init__(self, config: StandardValidatorConfig) -> None:
        super().__init__(config)
        self.config: StandardValidatorConfig = config

    @property
    def name(self) -> str:
        return self.config.name

    def validate(self, potential: Potential) -> ValidationMetrics:
        """
        Validate the potential.

        Args:
            potential: The potential to validate.

        Returns:
            ValidationMetrics: A model containing validation metrics and results.

        Raises:
            NotImplementedError: Always, as this is a placeholder.
        """
        logger.error("Standard Validator (Phonon/EOS/Elastic) is not yet implemented.")
        msg = "Standard Validator (Phonon/EOS/Elastic) is not yet implemented."
        raise NotImplementedError(msg)
