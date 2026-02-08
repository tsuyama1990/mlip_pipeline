from abc import abstractmethod

from mlip_autopipec.domain_models.config import ValidatorConfig
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.results import ValidationMetrics
from mlip_autopipec.interfaces.base_component import BaseComponent


class BaseValidator(BaseComponent[ValidatorConfig]):
    @property
    def name(self) -> str:
        return "validator"

    @abstractmethod
    def validate(self, potential: Potential) -> ValidationMetrics:
        """
        Validate potential and return metrics.

        Args:
            potential: The potential to validate.

        Returns:
            ValidationMetrics object containing validation results.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, config={self.config})>"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
